/// Implements the Quantum dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
