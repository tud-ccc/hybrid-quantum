#ifndef LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_
#define LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace qpu {

class SabreSwapAnalysis {
public:
    explicit SabreSwapAnalysis(Operation* moduleOp);
    ~SabreSwapAnalysis() = default;

    /// Main routing logic: rewrites gates using swaps/CNOTs
    LogicalResult apply(OpBuilder &builder);

    // Debug info: wrapper to expose coupling graph
    const llvm::SmallVector<std::pair<unsigned, unsigned>, 8> &
    getCouplingGraph() const
    {
        return coupling;
    }

    const llvm::SmallVector<unsigned, 16> &getInitialMapping() const
    {
        return mapping;
    }

private:
    void buildCouplingGraph(Operation* moduleOp);
    void collectTwoQubitGates(Operation* moduleOp);
    void computeDependencies();
    void computeFrontLayer();
    void computeDistanceMatrix();
    void generateInitialMapping();
    bool areAdjacent(unsigned p, unsigned q) const;

    unsigned numQubits = 0;
    // target-specific device topology
    llvm::SmallVector<std::pair<unsigned, unsigned>, 8> coupling;

    // logical-to-physical mapping
    llvm::SmallVector<unsigned, 16> mapping;

    // all two-qubit gates in program order
    llvm::SmallVector<Operation*, 32> gates;

    // gate deps (for potentially smarter SWAP policy)
    llvm::DenseMap<Operation*, llvm::SmallVector<Operation*, 4>> dependencies;

    // gates with no unresolved deps
    llvm::SmallVector<Operation*, 4> frontLayer;

    /// SSA tracking: logical qubit index → current Value
    llvm::DenseMap<Value, unsigned> valueToLogical;
    llvm::DenseMap<unsigned, Value> logicalToCurrentValue;

    // Floyd-Warshall distance matrix for all qubits
    llvm::SmallVector<llvm::SmallVector<unsigned, 16>, 16> distanceMatrix;

    // For each gate: list of its successor gates (needed to update front layer)
    llvm::DenseMap<Operation*, llvm::SmallVector<Operation*, 4>> successorMap;

    /// Heuristic cost function for a layer of gates
    unsigned computeHeuristicCost(
        const llvm::SmallVector<mlir::Operation*, 4> &layer,
        const llvm::SmallVector<unsigned, 16> &testMapping);
};

} // namespace qpu
} // namespace mlir

#endif // LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_
