/// Implements the QILLR dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::qillr;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QILLRDialect
//===----------------------------------------------------------------------===//

void QILLRDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.cpp.inc"
        >();
}
