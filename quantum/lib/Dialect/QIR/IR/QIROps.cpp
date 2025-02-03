/// Implements the QIR dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/QIR/IR/QIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "qir-ops"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/QIR/IR/QIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// QIRDialect
//===----------------------------------------------------------------------===//

void QIRDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/QIR/IR/QIROps.cpp.inc"
        >();
}

// parsers/printers
