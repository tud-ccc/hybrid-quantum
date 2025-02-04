/// Declares the Quantum passes.
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

/// Adds the quantum optimise pass patterns to @p patterns .
void populateQuantumOptimisePatterns(RewritePatternSet &patterns);

/// Constructs the lower-funnel-shift pass.
std::unique_ptr<Pass> createQuantumOptimisePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace quantum

} // namespace mlir
