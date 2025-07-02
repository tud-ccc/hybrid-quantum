/// Implements the QILLR dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"

#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"

using namespace mlir;
using namespace mlir::qillr;

//===- Generated implementation -------------------------------------------===//

#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// QILLRDialect
//===----------------------------------------------------------------------===//

void QILLRDialect::initialize()
{
    registerOps();
    registerTypes();
}
