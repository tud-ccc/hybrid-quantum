/// Implements the Quantum dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "quantum-types"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"
        >();
}
