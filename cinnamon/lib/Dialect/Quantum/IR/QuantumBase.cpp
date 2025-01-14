/// Implements the Quantum dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.h"

#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"


#define DEBUG_TYPE "quantum-base"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.cpp.inc"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize()
{
    registerOps();
    registerTypes();
    addAttributes<QuantumAxisAttr>();
}
