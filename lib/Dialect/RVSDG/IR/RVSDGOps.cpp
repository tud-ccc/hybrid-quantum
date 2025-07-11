/// Implements the RVSDG dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

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

#define DEBUG_TYPE "rvsdg-ops"

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RVSDGDialect
//===----------------------------------------------------------------------===//

void RVSDGDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.cpp.inc"
        >();
}
