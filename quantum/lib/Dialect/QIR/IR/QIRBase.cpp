/// Implements the QIR dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"

#define DEBUG_TYPE "qir-base"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/QIR/IR/QIRBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QIRDialect
//===----------------------------------------------------------------------===//

void QIRDialect::initialize()
{
    registerOps();
    registerTypes();
}
