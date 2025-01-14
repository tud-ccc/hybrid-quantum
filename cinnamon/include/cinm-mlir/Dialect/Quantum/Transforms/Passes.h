/// Declaration of the transform pass within Quantum dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quantum {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

} // namespace quantum
} // namespace mlir
