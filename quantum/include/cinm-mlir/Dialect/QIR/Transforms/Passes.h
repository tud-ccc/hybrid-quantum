/// Declaration of the transform pass within QIR dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace qir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/QIR/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

} // namespace qir
} // namespace mlir
