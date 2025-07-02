/// Declares the QPU passes.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/QPU/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qpu {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/QPU/Transforms/Passes.h.inc"

} // namespace qpu

} // namespace mlir
