//===- LinearUseAnalysis.cpp - Quantum Linear Property Analysis -----------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/Quantum/Analysis/LinearUseAnalysis.h"

#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "linear-use-analysis"

using namespace mlir;
using namespace mlir::quantum;
using namespace mlir::quantum::dataflow;

void LinearUseAnalysis::transferNoCloneOperation(
    Operation* op,
    const NoCloneLattice &before,
    NoCloneLattice* after)
{
    if (!before.isInitialized()) return;

    auto available = before.getAvailableQubits();

    // Consume qubit operands.
    for (OpOperand &operand : op->getOpOperands()) {
        Value value = operand.get();

        if (!isa<QubitType>(value.getType())) continue;

        // erase() returns 0 if the value was not available.
        // This also detects `%q, %q` within one operation.
        if (!available.erase(value)) violations.insert(&operand);
    }

    // Results establish new ownership.
    for (Value result : op->getResults())
        if (isa<QubitType>(result.getType())) available.insert(result);

    propagateIfChanged(after, after->join(available));
}

LogicalResult LinearUseAnalysis::visitOperation(
    Operation* op,
    const NoCloneLattice &before,
    NoCloneLattice* after)
{
    if (op->hasTrait<NoClone>()) {
        transferNoCloneOperation(op, before, after);
        return success();
    }

    // Ordinary operations don't affect quantum ownership.
    if (before.isInitialized())
        propagateIfChanged(after, after->join(before.getAvailableQubits()));

    return success();
}

void LinearUseAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call,
    mlir::dataflow::CallControlFlowAction action,
    const NoCloneLattice &before,
    NoCloneLattice* after)
{
    using Action = mlir::dataflow::CallControlFlowAction;

    switch (action) {
    case Action::ExternalCallee:
        // We intentionally treat calls atomically:
        //
        //   call operands -> consumed
        //   call results  -> available
        //
        // The callee is checked independently.
        assert(
            call->hasTrait<NoClone>()
            && "quantum call operation must have NoClone trait");

        transferNoCloneOperation(call.getOperation(), before, after);
        return;

    case Action::EnterCallee:
    case Action::ExitCallee:
        llvm_unreachable(
            "LinearUseAnalysis expects non-interprocedural call analysis");
    }
}

void LinearUseAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch,
    std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo,
    const NoCloneLattice &before,
    NoCloneLattice* after)
{
    if (!before.isInitialized()) return;

    auto available = before.getAvailableQubits();

    // Construct the source branch point.
    RegionBranchPoint source = RegionBranchPoint::parent();

    if (regionFrom) {
        Region &region = branch->getRegion(*regionFrom);

        // Sufficient for scf.if, scf.index_switch and scf.while.
        assert(
            llvm::hasSingleElement(region) && "expected single-block region");

        Operation* terminator = region.front().getTerminator();

        auto regionTerminator =
            cast<RegionBranchTerminatorOpInterface>(terminator);

        source = RegionBranchPoint(regionTerminator);
    }

    // Construct the destination.
    RegionSuccessor destination =
        regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
                 : RegionSuccessor(branch.getOperation());

    // Values flowing across this control-flow edge.
    OperandRange sourceValues =
        branch.getSuccessorOperands(source, destination);

    ValueRange destinationValues = branch.getSuccessorInputs(destination);

    assert(sourceValues.size() == destinationValues.size());

    for (auto [sourceValue, destinationValue] :
         llvm::zip_equal(sourceValues, destinationValues)) {

        if (!isa<QubitType>(destinationValue.getType())) continue;

        // Ownership is transferred, not copied.
        //
        // If sourceValue was live:
        //
        //     sourceValue -> destinationValue
        //
        // Otherwise destinationValue must not become live.
        if (available.erase(sourceValue)) available.insert(destinationValue);
    }

    propagateIfChanged(after, after->join(available));
}

ChangeResult NoCloneLattice::join(const AvailableSet &rhs)
{
    if (!available) {
        available = rhs;
        return ChangeResult::Change;
    }

    AvailableSet joined;
    for (Value value : *available)
        if (rhs.contains(value)) joined.insert(value);

    if (joined == *available) return ChangeResult::NoChange;

    available = std::move(joined);
    return ChangeResult::Change;
}

ChangeResult NoCloneLattice::join(const AbstractDenseLattice &rhs)
{
    const auto &other = static_cast<const NoCloneLattice &>(rhs);

    if (!other.isInitialized()) return ChangeResult::NoChange;

    return join(other.getAvailableQubits());
}

void LinearUseAnalysis::setToEntryState(NoCloneLattice* lattice)
{
    NoCloneLattice::AvailableSet available;

    auto* point = lattice->getAnchor().dyn_cast<mlir::ProgramPoint*>();

    assert(point && "dense lattice must be anchored at a program point");

    Block* block = point->getBlock();

    if (block && point->isBlockStart()) {
        for (BlockArgument arg : block->getArguments())
            if (isa<QubitType>(arg.getType())) available.insert(arg);
    }

    propagateIfChanged(lattice, lattice->join(available));
}

void NoCloneLattice::print(llvm::raw_ostream &os) const
{
    os << "available = ";

    if (!available) {
        os << "<uninitialized>";
        return;
    }

    os << "[";
    llvm::interleaveComma(*available, os, [&](mlir::Value value) {
        value.printAsOperand(os, mlir::OpPrintingFlags());
    });
    os << "]";
}
