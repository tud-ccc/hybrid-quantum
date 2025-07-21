/// Implements the SABRE swap analyses.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washimneupane@outlook.com)

#include "quantum-mlir/Dialect/QPU/Analysis/SabreSwapAnalysis.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

namespace mlir {
namespace qpu {

#define DEBUG_TYPE "SabreSwapAnalysis"

SabreSwapAnalysis::SabreSwapAnalysis(Operation* moduleOp)
{
    buildCouplingGraph(moduleOp);
    computeDistanceMatrix();
    generateInitialMapping();
    collectTwoQubitGates(moduleOp);
    computeDependencies();
    computeFrontLayer();

    // Assign logical ID to all allocOps. Maintain a bidirectional mapping.
    // Convert logicalToCurrent only in rewrite step
    unsigned nextLogicalId = 0;
    moduleOp->walk([&](Operation* op) {
        if (auto alloc = dyn_cast<quantum::AllocOp>(op)) {
            auto result = alloc.getResult();
            valueToLogical[result] = nextLogicalId;
            logicalToCurrentValue[nextLogicalId] = result;
            ++nextLogicalId;
        }
    });
}

void SabreSwapAnalysis::generateInitialMapping()
{
    mapping.resize(numQubits);
    for (unsigned i = 0; i < numQubits; ++i) mapping[i] = i;
}

void SabreSwapAnalysis::buildCouplingGraph(Operation* moduleOp)
{
    auto arrayAttr = moduleOp->getAttrOfType<ArrayAttr>("targets");
    if (!arrayAttr || arrayAttr.empty()) {
        moduleOp->emitWarning("qpu.module: no 'targets' attribute found");
        return;
    }
    auto target = mlir::dyn_cast<qpu::TargetAttr>(arrayAttr[0]);
    if (!target) {
        moduleOp->emitError("invalid element in 'targets'");
        return;
    }

    numQubits = target.getQubits().getValue().getZExtValue();
    auto cgArr = target.getCoupling();
    for (auto elem : cgArr.getValue()) {
        auto pair = mlir::cast<ArrayAttr>(elem);
        unsigned u = mlir::cast<IntegerAttr>(pair[0]).getValue().getZExtValue();
        unsigned v = mlir::cast<IntegerAttr>(pair[1]).getValue().getZExtValue();
        coupling.emplace_back(u, v);
        coupling.emplace_back(v, u);
    }
}

unsigned SabreSwapAnalysis::computeHeuristicCost(
    const llvm::SmallVector<Operation*, 4> &layer,
    const llvm::SmallVector<unsigned, 16> &testMapping)
{
    unsigned cost = 0;
    for (Operation* g : layer) {
        unsigned l0 = valueToLogical[g->getOperand(0)];
        unsigned l1 = valueToLogical[g->getOperand(1)];
        unsigned p0 = testMapping[l0], p1 = testMapping[l1];
        cost += distanceMatrix[p0][p1];
    }
    return cost;
}

// NOTE: This collects all two-qubit gates in program order
// Hoever, only CNOTs are emitted in the rewriter. So need to make sure only
// CNOTs are allowed in circuit until this is modified.
void SabreSwapAnalysis::collectTwoQubitGates(Operation* moduleOp)
{
    moduleOp->walk([&](Operation* op) {
        if (op->getNumOperands() == 2
            && isa<quantum::QubitType>(op->getOperand(0).getType())
            && isa<quantum::QubitType>(op->getOperand(1).getType()))
            gates.push_back(op);
    });
}

// Build dependency graph for all two-qubit gates
void SabreSwapAnalysis::computeDependencies()
{
    DenseMap<Value, Operation*> lastWriter;
    for (Operation* g : gates) {
        SmallVector<Operation*, 4> preds;

        for (Value v : g->getOperands()) {
            if (auto* prev = lastWriter.lookup(v)) {
                preds.push_back(prev);
                successorMap[prev].push_back(g);
            }
            lastWriter[v] = g;
        }
        dependencies[g] = preds;
    }
}

void SabreSwapAnalysis::computeFrontLayer()
{
    for (Operation* g : gates)
        if (dependencies[g].empty()) frontLayer.push_back(g);
}

// This was replaced with a distnace based adjacency check
// This is a naive implementation, but it works for small graphs
bool SabreSwapAnalysis::areAdjacent(unsigned p, unsigned q) const
{
    for (auto &e : coupling)
        if ((e.first == p && e.second == q) || (e.first == q && e.second == p))
            return true;
    return false;
}

void SabreSwapAnalysis::computeDistanceMatrix()
{

    // Set D[i][i] = 0 and D[i][j] = 1 if (i,j) is connected. Else infinite
    distanceMatrix.resize(
        numQubits,
        SmallVector<unsigned, 16>(
            numQubits,
            std::numeric_limits<unsigned>::max()));
    for (unsigned i = 0; i < numQubits; ++i) distanceMatrix[i][i] = 0;
    for (auto &[u, v] : coupling) distanceMatrix[u][v] = 1;

    // Floyd-Warshall matrix generation
    for (unsigned k = 0; k < numQubits; ++k) {
        for (unsigned i = 0; i < numQubits; ++i) {
            for (unsigned j = 0; j < numQubits; ++j) {
                if (distanceMatrix[i][k] != std::numeric_limits<unsigned>::max()
                    && distanceMatrix[k][j]
                           != std::numeric_limits<unsigned>::max()
                    && distanceMatrix[i][j]
                           > distanceMatrix[i][k] + distanceMatrix[k][j]) {
                    distanceMatrix[i][j] =
                        distanceMatrix[i][k] + distanceMatrix[k][j];
                }
            }
        }
    }
}

// TODO: Algorithm currently reads all 2 qubit gate into front layer, but only
// can emit CNOTs in rewriter. Need to make this generic for all 2 qubit ops
LogicalResult SabreSwapAnalysis::apply(OpBuilder &builder)
{
    DenseSet<Operation*> executed;
    int step = 0, MAX_STEPS = 1000; // safeguard against infinite loop

    while (!frontLayer.empty()) {
        if (++step > MAX_STEPS) {
            mlir::emitError(
                mlir::UnknownLoc::get(builder.getContext()),
                "SABRE routing failed: infinite loop");
            return failure();
        }

        // Try to find an adjacent gate (distance == 1)
        Operation* executableGate = nullptr;
        for (auto* g : frontLayer) {
            unsigned l0 = valueToLogical[g->getOperand(0)];
            unsigned l1 = valueToLogical[g->getOperand(1)];
            unsigned p0 = mapping[l0], p1 = mapping[l1];
            if (distanceMatrix[p0][p1] == 1) {
                executableGate = g;
                break;
            }
        }

        if (executableGate) {
            Value op0 = executableGate->getOperand(0);
            Value op1 = executableGate->getOperand(1);
            unsigned l0 = valueToLogical[op0], l1 = valueToLogical[op1];
            Value cur0 = logicalToCurrentValue[l0];
            Value cur1 = logicalToCurrentValue[l1];
            builder.setInsertionPoint(executableGate);

            // TODO: Emits CNOT. Modify to handle generic two-qubit gates
            auto newCNOT = builder.create<quantum::CNOTOp>(
                executableGate->getLoc(),
                cur0,
                cur1);

            // SSA: update current values for logical indices
            logicalToCurrentValue[l0] = newCNOT.getResult(0);
            logicalToCurrentValue[l1] = newCNOT.getResult(1);

            // Redirect all uses and erase
            for (unsigned i = 0; i < executableGate->getNumResults(); ++i)
                executableGate->getResult(i).replaceAllUsesWith(
                    newCNOT.getResult(i));

            executed.insert(executableGate);
            executableGate->erase();

            // Update front layer
            frontLayer.erase(
                std::remove(
                    frontLayer.begin(),
                    frontLayer.end(),
                    executableGate),
                frontLayer.end());
            for (Operation* succ : successorMap[executableGate]) {
                bool ready = llvm::all_of(
                    dependencies[succ],
                    [&](Operation* pred) { return executed.contains(pred); });
                if (ready) frontLayer.push_back(succ);
            }

        } else {
            // Swap heuristic: minimize sum of distances in frontLayer
            unsigned bestL0 = 0, bestL1 = 1;
            unsigned bestCost = UINT_MAX;

            for (Operation* g : frontLayer) {
                unsigned l0 = valueToLogical[g->getOperand(0)];
                unsigned l1 = valueToLogical[g->getOperand(1)];

                for (unsigned origL : {l0, l1}) {
                    unsigned origP = mapping[origL];

                    for (auto &[u, v] : coupling) {
                        if (u != origP) continue;
                        unsigned neighborP = v;

                        // Find logical qubit currently mapped to neighborP
                        unsigned swapWithL = UINT_MAX;
                        for (unsigned j = 0; j < mapping.size(); ++j)
                            if (mapping[j] == neighborP) {
                                swapWithL = j;
                                break;
                            }

                        if (swapWithL == UINT_MAX || origL == swapWithL)
                            continue;

                        auto candidate = mapping;
                        std::swap(candidate[origL], candidate[swapWithL]);

                        // Compute sum of distances heuristics over frontLayer
                        // from sabre paper
                        unsigned cost = 0;
                        for (Operation* fg : frontLayer) {
                            unsigned fl0 = valueToLogical[fg->getOperand(0)];
                            unsigned fl1 = valueToLogical[fg->getOperand(1)];
                            unsigned fp0 = candidate[fl0];
                            unsigned fp1 = candidate[fl1];
                            cost += distanceMatrix[fp0][fp1];
                        }

                        if (cost < bestCost) {
                            bestCost = cost;
                            bestL0 = origL;
                            bestL1 = swapWithL;
                        }
                    }
                }
            }

            // Perform best SWAP (always a SWAP, never a generic two-qubit gate)
            Value cur0 = logicalToCurrentValue[bestL0];
            Value cur1 = logicalToCurrentValue[bestL1];
            builder.setInsertionPoint(frontLayer.front());
            auto swapOp = builder.create<quantum::SWAPOp>(
                frontLayer.front()->getLoc(),
                cur0,
                cur1);
            logicalToCurrentValue[bestL0] = swapOp.getResult(1);
            logicalToCurrentValue[bestL1] = swapOp.getResult(0);
            std::swap(mapping[bestL0], mapping[bestL1]);
        }
    }

    return success();
}

} // namespace qpu
} // namespace mlir
