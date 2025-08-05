/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTAttributes.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTTypes.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Record.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

#define DEBUG_TYPE "qqt-ops"

using namespace mlir;
using namespace mlir::qqt;

//===----------------------------------------------------------------------===//
// Printer / Parser
//===----------------------------------------------------------------------===//

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QQTDialect
//===----------------------------------------------------------------------===//

void QQTDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.cpp.inc"
        >();
}
