/// Declaration of the Quantum dialect types.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>


namespace mlir {
namespace quantum {
class QubitTypeStorage : public TypeStorage {
public:
  using KeyTy = unsigned;

  QubitTypeStorage(unsigned size) : size(size) {}

  bool operator==(const KeyTy &key) const { return key == size; }

  static QubitTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key);

  unsigned size;
};

class QubitType : public Type::TypeBase<QubitType, Type, QubitTypeStorage> {
public:
  using Base::Base;

  static QubitType get(MLIRContext *context, unsigned size);

  unsigned getSize() const;
};

} // namespace Quantum
} // namespace mlir


//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.h.inc"

//===----------------------------------------------------------------------===//

