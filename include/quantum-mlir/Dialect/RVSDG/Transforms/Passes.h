/// Declares the RVSDG passes.
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/RVSDG/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace rvsdg {

std::unique_ptr<Pass> createControlFlowHoistingPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/RVSDG/Transforms/Passes.h.inc"

} // namespace rvsdg

} // namespace mlir
