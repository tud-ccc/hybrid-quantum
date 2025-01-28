/// Implements the Quantum dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-types"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// QubitTypeStorage Implementation
//===----------------------------------------------------------------------===//
QubitTypeStorage *QubitTypeStorage::construct(TypeStorageAllocator &allocator, const KeyTy &key) {
  return new (allocator.allocate<QubitTypeStorage>()) QubitTypeStorage(key);
}

//===----------------------------------------------------------------------===//
// QubitType Implementation
//===----------------------------------------------------------------------===//

QubitType QubitType::get(MLIRContext *context, unsigned size) {
  return Base::get(context, size);
}

unsigned QubitType::getSize() const {
  return getImpl()->size;
}
//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//
void QuantumDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"
      >();
}




