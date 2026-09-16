//===--- IntervalAnalysis.h - Quantum Register Interval Anaysis --*- C++-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_ANALYSIS_DATAFLOW_INTERVALANALYSIS_H
#define QUANTUM_MLIR_ANALYSIS_DATAFLOW_INTERVALANALYSIS_H

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"

#include <mlir/Analysis/DataFlow/IntegerRangeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

namespace quantum {
namespace dataflow {

//===----------------------------------------------------------------------===//
// RegisterRangesLattice
//===----------------------------------------------------------------------===//
class RegisterRangesLattice : public mlir::dataflow::Lattice<RegisterRanges> {
public:
    using mlir::dataflow::Lattice<RegisterRanges>::Lattice;
};

//===----------------------------------------------------------------------===//
// RegisterRangesAnalysis
//===----------------------------------------------------------------------===//
class RegisterRangesAnalysis
        : public mlir::dataflow::SparseForwardDataFlowAnalysis<
              RegisterRangesLattice> {
public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    /// At an entry point, we cannot reason about integer value ranges.
    void setToEntryState(RegisterRangesLattice* lattice) override
    { propagateIfChanged(lattice, lattice->join(RegisterRanges())); }

    /// Visit an operation. Invoke the transfer function on each operation that
    /// implements `InferIntRangeInterface`.
    LogicalResult visitOperation(
        Operation* op,
        ArrayRef<const RegisterRangesLattice*> operands,
        ArrayRef<RegisterRangesLattice*> results) override;

    /// Visit block arguments or operation results of an operation with region
    /// control-flow for which values are not defined by region control-flow.
    /// This function calls `InferIntRangeInterface` to provide values for block
    /// arguments or tries to reduce the range on loop induction variables with
    /// known bounds.
    // void visitNonControlFlowArguments(
    //     Operation* op,
    //     const RegionSuccessor &successor,
    //     ArrayRef<RegisterRangesLattice*> argLattices,
    //     unsigned firstIndex) override;
};

} // namespace dataflow
} // namespace quantum
} // namespace mlir

#endif // QUANTUM_MLIR_ANALYSIS_DATAFLOW_INTERVALANALYSIS_H
