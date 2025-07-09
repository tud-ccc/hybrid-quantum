/// Implements the QPU dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::qpu;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

LogicalResult
QPUDialect::verifyOperationAttribute(mlir::Operation*, mlir::NamedAttribute)
{
    // TODO
    return success();
}

//===----------------------------------------------------------------------===//
// QPUDialect
//===----------------------------------------------------------------------===//

void QPUDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.cpp.inc"
        >();
}
