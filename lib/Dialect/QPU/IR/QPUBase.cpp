/// Implements the QPU dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"

#include "quantum-mlir/Dialect/QPU/IR/QPU.h"

#define DEBUG_TYPE "qpu-base"

using namespace mlir;
using namespace mlir::qpu;

//===- Generated implementation -------------------------------------------===//

#define GET_DEFAULT_TYPE_PRINTER_PARSER
#define GET_DEFAULT_ATTR_PRINTER_PARSER
#include "quantum-mlir/Dialect/QPU/IR/QPUBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// QPUDialect
//===----------------------------------------------------------------------===//

void QPUDialect::initialize()
{
    registerOps();
    registerTypes();
    registerAttributes();
}
