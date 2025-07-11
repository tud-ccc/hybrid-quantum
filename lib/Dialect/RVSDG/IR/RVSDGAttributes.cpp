/// Implements the RVSDG dialect attributes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RVSDGDialect
//===----------------------------------------------------------------------===//

void RVSDGDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.cpp.inc"
        >();
}
