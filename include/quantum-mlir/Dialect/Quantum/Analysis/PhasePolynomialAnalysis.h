//===--- PhasePolynomialAnalysis.h - Quantum Phase Poly Anaysis --*- C++-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_ANALYSIS_DATAFLOW_PHASEPOLYNOMIALANALYSIS_H
#define QUANTUM_MLIR_ANALYSIS_DATAFLOW_PHASEPOLYNOMIALANALYSIS_H

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h"

#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir {

namespace quantum {
namespace dataflow {

//===----------------------------------------------------------------------===//
// PhasePolynomialLattice
//===----------------------------------------------------------------------===//
class PhasePolynomialLattice : public mlir::dataflow::Lattice<PhasePolynomial> {
public:
    using mlir::dataflow::Lattice<PhasePolynomial>::Lattice;
};

//===----------------------------------------------------------------------===//
// PhasePolynomialAnalysis
//===----------------------------------------------------------------------===//
class PhasePolynomialAnalysis
        : public mlir::dataflow::SparseForwardDataFlowAnalysis<
              PhasePolynomialLattice> {
public:
    explicit PhasePolynomialAnalysis(DataFlowSolver &solver)
            : SparseForwardDataFlowAnalysis(solver)
    {}

    /// At an entry point, we cannot reason about phase polynomials.
    void setToEntryState(PhasePolynomialLattice* lattice) override
    { propagateIfChanged(lattice, lattice->join(PhasePolynomial())); }

    /// Visit an operation. Invoke the transfer function on each operation that
    /// implements `InferPhasePolynomialInterface`.
    LogicalResult visitOperation(
        Operation* op,
        ArrayRef<const PhasePolynomialLattice*> operands,
        ArrayRef<PhasePolynomialLattice*> results) override;
};
} // namespace dataflow
} // namespace quantum

} // namespace mlir

#endif // QUANTUM_MLIR_ANALYSIS_DATAFLOW_PHASEPOLYNOMIALANALYSIS_H
