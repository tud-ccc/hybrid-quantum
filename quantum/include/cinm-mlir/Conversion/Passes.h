/// Declaration of the conversion pass within Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"
#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

namespace mlir::quantum {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::quantum