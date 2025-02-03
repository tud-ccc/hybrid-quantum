/// Implements the QIR dialect types.
///
/// @file

#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "qir-types"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QIRDialect
//===----------------------------------------------------------------------===//

void QIRDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.cpp.inc"
        >();
}
