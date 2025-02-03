/// Implements the QIR dialect base.
///
/// @file

#include "cinm-mlir/Dialect/QIR/IR/QIRBase.h"

#include "cinm-mlir/Dialect/QIR/IR/QIRDialect.h"

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
