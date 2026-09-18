//===--- LinearUseAnalysis.h - Quantum Linear Use Anaysis ---------*-C++-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir {
namespace quantum {
namespace dataflow {

struct LinearUseState {
    bool initialized = false;
    DenseSet<Value> live;
};

//===----------------------------------------------------------------------===//
// NoCloneLattice
//===----------------------------------------------------------------------===//
class NoCloneLattice : public mlir::dataflow::AbstractDenseLattice {
public:
    using AvailableSet = llvm::SmallDenseSet<Value, 8>;

    using AbstractDenseLattice::AbstractDenseLattice;

    bool isInitialized() const { return available.has_value(); }

    const AvailableSet &getAvailableQubits() const
    {
        assert(available);
        return *available;
    }

    ChangeResult join(const AvailableSet &rhs);

    ChangeResult join(const AbstractDenseLattice &rhs) override;

    void print(llvm::raw_ostream &os) const override;

private:
    std::optional<AvailableSet> available;
};

//===----------------------------------------------------------------------===//
// LinearUseAnalysis
//===----------------------------------------------------------------------===//

class LinearUseAnalysis
        : public mlir::dataflow::DenseForwardDataFlowAnalysis<NoCloneLattice> {
public:
    using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

    void setToEntryState(NoCloneLattice* lattice) override;

    LogicalResult visitOperation(
        Operation* op,
        const NoCloneLattice &before,
        NoCloneLattice* after) override;

    void visitCallControlFlowTransfer(
        CallOpInterface call,
        mlir::dataflow::CallControlFlowAction action,
        const NoCloneLattice &before,
        NoCloneLattice* after) override;

    void visitRegionBranchControlFlowTransfer(
        RegionBranchOpInterface branch,
        std::optional<unsigned> regionFrom,
        std::optional<unsigned> regionTo,
        const NoCloneLattice &before,
        NoCloneLattice* after) override;

    void transferNoCloneOperation(
        Operation* op,
        const NoCloneLattice &before,
        NoCloneLattice* after);

    bool hasViolations() const { return !violations.empty(); }

    const llvm::SmallSetVector<OpOperand*, 8> &getViolations() const
    { return violations; }

private:
    llvm::SmallSetVector<OpOperand*, 8> violations;
};

} // namespace dataflow
} // namespace quantum
} // namespace mlir
