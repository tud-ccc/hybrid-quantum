/// Declaration of the conversion pass within Quantum dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/QuantumToLLVM/QuantumToLLVM.h"
#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"


namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/QuantumPasses.h.inc"

//===----------------------------------------------------------------------===//

} //namespace mlir