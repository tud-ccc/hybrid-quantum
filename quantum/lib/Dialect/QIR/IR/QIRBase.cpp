/// Implements the QIR dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QIR/IR/QIRBase.h"

#include "quantum-mlir/Dialect/QIR/IR/QIR.h"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#include "quantum-mlir/Dialect/QIR/IR/QIRBase.cpp.inc"

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
