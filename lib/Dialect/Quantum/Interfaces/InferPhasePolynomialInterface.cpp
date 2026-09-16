//===- InferPhasePolynomialInterface.cpp -  Phase Polynomial interface ---===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h"

#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.cpp.inc"

#include <algorithm>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace mlir::quantum;

void mlir::quantum::phasepolynomial::detail::defaultInferResultPolynomial(
    InferPhasePolynomialInterface interface,
    ArrayRef<PhasePolynomial> argRanges,
    SetPolynomialFn setResultPolynomials)
{
    // Standard implementation passes for each input operand its analysis result
    // to the corresponding result value
    for (auto &&[result, polynomial] :
         llvm::zip(interface->getResults(), argRanges))
        if (llvm::isa<QubitType>(result.getType()))
            setResultPolynomials(result, polynomial);
}

bool ConstantPhasePolynomial::operator==(
    const ConstantPhasePolynomial &other) const
{ return getParity() == other.getParity() && getEpoch() == other.getEpoch(); }

const llvm::BitVector &ConstantPhasePolynomial::getParity() const
{ return parityVal; }

unsigned ConstantPhasePolynomial::getEpoch() const { return epochVal; }

raw_ostream &mlir::quantum::operator<<(
    raw_ostream &os,
    const ConstantPhasePolynomial &polynomial)
{
    os << polynomial.getEpoch() << " @ [";

    for (size_t i = 0; i < polynomial.getParity().size(); ++i)
        os << (polynomial.getParity()[i] ? "1" : "0");

    os << "]";
    return os;
}

raw_ostream &
mlir::quantum::operator<<(raw_ostream &os, const PhasePolynomial &polynomial)
{
    polynomial.print(os);
    return os;
}

ConstantPhasePolynomial
ConstantPhasePolynomial::parityOr(const ConstantPhasePolynomial &other) const
{
    unsigned maxEpoch = std::max(getEpoch(), other.getEpoch());
    unsigned maxCount = std::max(getParity().size(), other.getParity().size());
    ConstantPhasePolynomial result(maxCount, maxEpoch);
    result.parityVal = llvm::BitVector(getParity());
    result.parityVal |= other.getParity();
    return result;
}

ConstantPhasePolynomial
ConstantPhasePolynomial::parityAnd(const ConstantPhasePolynomial &other) const
{
    unsigned maxEpoch = std::max(getEpoch(), other.getEpoch());
    unsigned maxCount = std::max(getParity().size(), other.getParity().size());
    ConstantPhasePolynomial result(maxCount, maxEpoch);
    result.parityVal = llvm::BitVector(getParity());
    result.parityVal &= other.getParity();
    return result;
}

ConstantPhasePolynomial
ConstantPhasePolynomial::parityXor(const ConstantPhasePolynomial &other) const
{
    unsigned maxEpoch = std::max(getEpoch(), other.getEpoch());
    unsigned maxCount = std::max(getParity().size(), other.getParity().size());
    ConstantPhasePolynomial result(maxCount, maxEpoch);
    result.parityVal = llvm::BitVector(getParity());
    result.parityVal ^= other.getParity();
    return result;
}
