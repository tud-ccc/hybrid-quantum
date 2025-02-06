/// Implements the Quantum dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.h"

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
