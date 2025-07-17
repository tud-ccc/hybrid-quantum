/// Implements the RVSDG dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"

#define DEBUG_TYPE "rvsdg-base"

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated implementation -------------------------------------------===//

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// RVSDGDialect
//===----------------------------------------------------------------------===//

void RVSDGDialect::initialize()
{
    registerOps();
    registerTypes();
    registerAttributes();
}
