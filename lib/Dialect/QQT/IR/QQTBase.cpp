/// Implements the QQT dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QQT/IR/QQTBase.h"

#include "quantum-mlir/Dialect/QQT/IR/QQT.h"

#define DEBUG_TYPE "qqt-base"

using namespace mlir;
using namespace mlir::qqt;

//===- Generated implementation -------------------------------------------===//

#define GET_DEFAULT_TYPE_PRINTER_PARSER
#define GET_DEFAULT_ATTR_PRINTER_PARSER
#include "quantum-mlir/Dialect/QQT/IR/QQTBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// QQTDialect
//===----------------------------------------------------------------------===//

void QQTDialect::initialize()
{
    registerOps();
    registerTypes();
    registerAttributes();
}
