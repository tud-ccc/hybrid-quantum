/// Declares the QILLR passes.
///
/// @file
/// @author     Washim Neupane (washimneupane@outlook.com)
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qillr {

/// Adds the quantum optimise pass patterns to @p patterns .
void populateUGatesDecompositionPatterns(RewritePatternSet &patterns);

/// Constructs the lower-funnel-shift pass.
std::unique_ptr<Pass> createDecomposeUGatesPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h.inc"

} // namespace qillr

} // namespace mlir
