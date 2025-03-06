/// Implements the QIR dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/QIR/IR/QIRBase.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/QIR/IR/QIRBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// QIRDialect
//===----------------------------------------------------------------------===//

void QIRDialect::initialize()
{
    registerOps();
    registerTypes();
}
