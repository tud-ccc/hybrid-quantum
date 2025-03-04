/// Implements the QIR dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

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