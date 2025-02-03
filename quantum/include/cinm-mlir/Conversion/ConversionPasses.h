/// Declaration of the conversion pass within Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

namespace mlir::quantum {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/ConversionPasses.h.inc"

//===----------------------------------------------------------------------===//

} //namespace mlir::quantum

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

namespace mlir::qir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/ConversionPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::qir