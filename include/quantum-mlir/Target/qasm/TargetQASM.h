//===- TargetQASM.h - QILLR to OpenQASM Translation -----------------------===//
//
// A translator that handles QILLR dialect ops for OpenQASM 2.0.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Value.h"

namespace mlir {
namespace qillr {

LogicalResult QILLRTranslateToQASM(Operation* op, raw_ostream &os);

void registerQILLRToOpenQASMTranslation();
} // namespace qillr
} // namespace mlir
