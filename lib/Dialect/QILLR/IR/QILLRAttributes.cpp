/// Implements the QILLR dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/IR/QILLRAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::qillr;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "quantum-mlir/Dialect/QILLR/IR/QILLRAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
