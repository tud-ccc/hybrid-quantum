//===- InferPhasePolynomialInterface.h - Phase Poly. Inference ---*- C++-*-===//
//
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the phase polynomial inference interface
// defined in `InferPhasePolynomialInterface.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H
#define MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include <cstddef>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace quantum {

class ConstantPhasePolynomial {
public:
    ConstantPhasePolynomial(const unsigned qubitCount, const unsigned epoch = 0)
            : parityVal(qubitCount, false),
              epochVal(epoch)
    {}

    bool operator==(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityOr(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityAnd(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityXor(const ConstantPhasePolynomial &other) const;

    const llvm::BitVector &getParity() const;

    void reset() { parityVal.reset(); }

    void setBit(size_t index) { parityVal.set(index); }

    void setBit(size_t start, size_t end) { parityVal.set(start, end); }

    unsigned getEpoch() const;

    void setEpoch(unsigned epoch) { epochVal = epoch; }

    friend llvm::raw_ostream &operator<<(
        llvm::raw_ostream &os,
        const ConstantPhasePolynomial &polynomial);

    friend llvm::hash_code hash_value(const ConstantPhasePolynomial &p)
    {
        auto h = llvm::hash_value(p.getEpoch());

        for (auto word : p.getParity().getData())
            h = llvm::hash_combine(h, word);

        return static_cast<unsigned>(h);
    }

private:
    llvm::BitVector parityVal;
    unsigned epochVal;
};

llvm::raw_ostream &
operator<<(llvm::raw_ostream &, const ConstantPhasePolynomial &);

/// This lattice value represents the phase polynomial of an SSA value.
/// Since !quantum.qubit<N> may be an N-valued qubit register it holds
/// N polynomials.
class PhasePolynomial {
public:
    PhasePolynomial() = default;

    /// Create a phase polynomial lattice value
    PhasePolynomial(ConstantPhasePolynomial value, size_t qubit)
    {
        values.push_back(value);
        qubitPos.push_back(qubit);
    }

    PhasePolynomial(
        llvm::SmallVector<ConstantPhasePolynomial> vals,
        llvm::SmallVector<size_t> qubits)
    {
        values.append(vals.begin(), vals.end());
        qubitPos.append(qubits.begin(), qubits.end());
    }

    /// Check whether the state is uninitialized
    bool isUninitialized() const { return values.empty(); }

    llvm::SmallVector<size_t> getQubit() const { return qubitPos; }

    /// Get the known phase polynomial.
    const llvm::SmallVector<ConstantPhasePolynomial> &getValue() const
    {
        assert(!isUninitialized());
        return values;
    }

    /// Compare two phase polynomials.
    bool operator==(const PhasePolynomial &rhs) const
    { return values == rhs.values; }

    /// Print the phase polynomial
    void print(llvm::raw_ostream &os) const
    {
        for (ConstantPhasePolynomial c : values) os << c;
    }

    /// Compute the combination of two phase polynomials
    static PhasePolynomial
    join(const PhasePolynomial &lhs, const PhasePolynomial &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;

        assert(lhs.qubitPos == rhs.qubitPos);

        llvm::SmallVector<ConstantPhasePolynomial> joinedVals;
        for (auto &&[lval, rval] : llvm::zip_equal(lhs.values, rhs.values))
            joinedVals.emplace_back(lval.parityOr(rval));

        return PhasePolynomial(joinedVals, lhs.qubitPos);
    }

    /// Compute the symmetric difference of two phase polynomials
    static PhasePolynomial
    meet(const PhasePolynomial &lhs, const PhasePolynomial &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;

        llvm::SmallVector<ConstantPhasePolynomial> meetVals;
        for (auto &&[lval, rval] : llvm::zip_equal(lhs.values, rhs.values))
            meetVals.emplace_back(lval.parityXor(rval));

        return PhasePolynomial(meetVals, rhs.qubitPos);
    }

private:
    /// The known phase polynomials.
    llvm::SmallVector<ConstantPhasePolynomial> values;
    llvm::SmallVector<size_t> qubitPos;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const PhasePolynomial &);

using SetPolynomialFn =
    llvm::function_ref<void(Value, const PhasePolynomial &)>;

class InferPhasePolynomialInterface;

namespace phasepolynomial::detail {

void defaultInferResultPolynomial(
    InferPhasePolynomialInterface interface,
    ArrayRef<PhasePolynomial> argPolynomials,
    SetPolynomialFn setResultPolynomials);

} // namespace phasepolynomial::detail

} // namespace quantum
} // namespace mlir

namespace llvm {

template<>
struct DenseMapInfo<llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial>> {
    static llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial>
    getEmptyKey()
    {
        return llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial>{
            mlir::quantum::ConstantPhasePolynomial(0, ~0u)};
    }

    static llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial>
    getTombstoneKey()
    {
        return llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial>{
            mlir::quantum::ConstantPhasePolynomial(0, ~0u - 1)};
    }

    static unsigned getHashValue(
        const llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial> &v)
    {
        auto h = llvm::hash_combine_range(v.begin(), v.end());
        return static_cast<unsigned>(h);
    }

    static bool isEqual(
        const llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial> &lhs,
        const llvm::SmallVector<mlir::quantum::ConstantPhasePolynomial> &rhs)
    { return lhs == rhs; }
};

} // namespace llvm

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h.inc"

#endif // MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H
