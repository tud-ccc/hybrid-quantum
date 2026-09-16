/// Declares the Quantum passes.
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

void populateQuantumOptimisePatterns(RewritePatternSet &patterns);

void populateHermitianCancelPatterns(RewritePatternSet &patterns);

void populateScfToRVSDGPatterns(
    TypeConverter converter,
    RewritePatternSet &patterns);

/// Constructs the lower-funnel-shift pass.
std::unique_ptr<Pass> createQuantumOptimisePass();

/// Pass that realizes self-adjoint gate cancellation
std::unique_ptr<Pass> createHermitianCancelPass();

std::unique_ptr<Pass> createScfToRVSDGPass();

std::unique_ptr<Pass> createPhasePolynomialMergePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace quantum

} // namespace mlir
