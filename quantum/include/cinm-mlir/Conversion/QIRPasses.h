/// Declaration of the conversion pass within QIR dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/QIRPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir