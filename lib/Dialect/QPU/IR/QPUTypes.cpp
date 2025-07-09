/// Implements the QPU dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/ExtensibleDialect.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "qpu-types"

using namespace mlir;
using namespace mlir::qpu;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPUDialect
//===----------------------------------------------------------------------===//

void QPUDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.cpp.inc"
        >();
}
