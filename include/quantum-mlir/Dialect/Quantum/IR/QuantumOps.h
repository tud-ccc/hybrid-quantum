/// Declaration of the Quantum dialect ops.
///
/// @file

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace quantum {} // namespace quantum

namespace OpTrait {
template<typename ConcreteType>
class Unitary : public TraitBase<ConcreteType, Unitary> {};
template<typename ConcreteType>
class Hermitian : public TraitBase<ConcreteType, Hermitian> {};
template<typename ConcreteType>
class Kernel : public TraitBase<ConcreteType, Kernel> {};

} // namespace OpTrait

} // namespace mlir

//===- Generated includes
//-------------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h.inc"

// namespace mlir::Quantum
//===----------------------------------------------------------------------===//