/// Declaration of the Quantum dialect ops.
///
/// @file

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <llvm/Support/Casting.h>

namespace mlir {
namespace quantum {

void buildTerminatedBody(
    OpBuilder &builder,
    Location loc,
    Value condition,
    ValueRange capturedArgs);

template<typename ConcreteType>
class NoClone : public OpTrait::TraitBase<ConcreteType, NoClone> {
public:
    static LogicalResult verifyTrait(Operation* op);
};

template<typename ConcreteType>
class Hermitian : public OpTrait::TraitBase<ConcreteType, Hermitian> {
public:
    /// Override the 'foldTrait' hook to support trait based folding on the
    /// concrete operation.
    static LogicalResult foldTrait(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &results);
};

template<typename ConcreteType>
class Unitary : public OpTrait::TraitBase<ConcreteType, Unitary> {
public:
    /// Override the 'foldTrait' hook to support trait based folding on the
    /// concrete operation.
    static LogicalResult foldTrait(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &results);
};

template<typename AdjointToType>
class AdjointTo {
public:
    template<typename ConcreteType>
    struct Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
        static constexpr ::llvm::StringLiteral getAdjointOperationName()
        {
            return ConcreteType::getOperationName();
        }
    };
};

} // namespace quantum

} // namespace mlir

//===----------------------------------------------------------------------===//
//===- Generated includes
//-------------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h.inc"

// namespace mlir::Quantum
//===----------------------------------------------------------------------===//
