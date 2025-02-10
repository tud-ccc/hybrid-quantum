/// Implements the QIR dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/QIR/IR/QIROps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

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