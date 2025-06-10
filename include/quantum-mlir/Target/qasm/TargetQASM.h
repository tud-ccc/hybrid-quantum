//===- TargetQASM.h - QIR to OpenQASM Translation -------------------------===//
//
// A translator that handles QIR dialect ops for OpenQASM 2.0.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Value.h"

namespace mlir {
namespace qir {

LogicalResult QIRTranslateToQASM(Operation* op, raw_ostream &os);

void registerQIRToOpenQASMTranslation();
} // namespace qir
} // namespace mlir
