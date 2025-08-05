/// Implements the QQT dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QQT/IR/QQTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::qqt;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "quantum-mlir/Dialect/QQT/IR/QQTAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QQTDialect
//===----------------------------------------------------------------------===//

void QQTDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "quantum-mlir/Dialect/QQT/IR/QQTAttributes.cpp.inc"
        >();
}
