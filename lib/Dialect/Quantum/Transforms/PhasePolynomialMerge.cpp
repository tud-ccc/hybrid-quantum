/// Implements rotation gate angel merge based on phase polynomial analysis.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h"
#include "quantum-mlir/Dialect/Quantum/Analysis/PhasePolynomialAnalysis.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h"
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#define DEBUG_TYPE "phase-poly-merge"

using namespace mlir;
using namespace mlir::quantum;
using namespace mlir::quantum::dataflow;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_PHASEPOLYNOMIALMERGE
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct PhasePolynomialMergePass
        : quantum::impl::PhasePolynomialMergeBase<PhasePolynomialMergePass> {
    using PhasePolynomialMergeBase::PhasePolynomialMergeBase;

    void runOnOperation() override;
};

static Value buildSum(ArrayRef<Value> thetas, OpBuilder &builder, Location loc)
{
    if (thetas.size() == 1) return thetas[0];

    SmallVector<Value> next;
    for (size_t i = 0; i < thetas.size(); i += 2) {
        if (i + 1 < thetas.size()) {
            next.push_back(builder.createOrFold<arith::AddFOp>(
                loc,
                thetas[i].getType(),
                thetas[i],
                thetas[i + 1]));
        } else {
            next.push_back(thetas[i]);
        }
    }
    return buildSum(next, builder, loc);
}

} // namespace

void PhasePolynomialMergePass::runOnOperation()
{
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<quantum::dataflow::PhasePolynomialAnalysis>();

    llvm::DenseMap<SmallVector<ConstantPhasePolynomial>, SmallVector<RzOp>>
        groups;

    // Each qpu::circuit is the root of a quantum circuit
    // Run analysis on each circuit and collect operations with same-valued
    // epochs and parity bits
    getOperation().walk([&](qpu::CircuitOp circOp) {
        // Workaround: assign each allocop a position
        size_t nqubits = 0;
        circOp->walk([&](quantum::AllocOp allocOp) {
            allocOp.setPosAttr(
                IntegerAttr::get(
                    IntegerType::get(allocOp.getContext(), 32),
                    nqubits));
            nqubits += allocOp.getResult().getType().getSize();
        });

        if (failed(solver.initializeAndRun(circOp))) return signalPassFailure();

        LLVM_DEBUG(
            llvm::dbgs() << "[PhasePolyMerge]: Collect groups for circuit: "
                         << circOp.getSymName() << "\n");

        circOp->walk([&](RzOp rzop) {
            const PhasePolynomialLattice* lattice =
                solver.lookupState<PhasePolynomialLattice>(rzop.getInput());
            if (!lattice)
                rzop->emitOpError(
                    "does not possess a phase polynomial "
                    "representation.");

            LLVM_DEBUG(
                llvm::dbgs() << "[PhasePolyMerge]: Collect "
                             << lattice->getValue() << " for " << rzop << "\n");

            auto key = lattice->getValue().getValue();
            groups[key].push_back(rzop);
        });
    });

    // Merge all rotations that share the same parity bits inside the same epoch
    for (auto &[poly, ops] : groups) {
        if (ops.size() < 2) {
            LLVM_DEBUG(
                llvm::dbgs() << "[PhasePolyMerge]: Ignore operation " << ops[0]
                             << " with " << poly[0] << "\n");
            continue;
        }

        OpBuilder builder(ops[0]);
        SmallVector<Value> thetas;
        for (RzOp rz : ops) thetas.push_back(rz.getTheta());

        Value accumulatedTheta = buildSum(thetas, builder, ops[0]->getLoc());

        RzOp mergedRz = RzOp::create(
            builder,
            ops[0].getLoc(),
            ops[0].getInput(),
            accumulatedTheta);

        ops[0].getResult().replaceAllUsesWith(mergedRz.getResult());
        for (RzOp rz : ops) {
            rz.getResult().replaceAllUsesWith(rz.getInput());
            rz->erase();
        }
    }
}

std::unique_ptr<Pass> mlir::quantum::createPhasePolynomialMergePass()
{
    return std::make_unique<PhasePolynomialMergePass>();
}
