/// Declares the Quantum passes.
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/Transforms/OneToNTypeConversion.h>

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

/// Adds the quantum optimise pass patterns to @p patterns .
void populateQuantumOptimisePatterns(RewritePatternSet &patterns);

/// Adds the legalization pass patterns to @p patterns .
void populateMultiQubitLegalizationPatterns(
    TypeConverter converter,
    RewritePatternSet &patterns);

/// Adds the scf-to-rvsdg pass patterns to @p patterns .
void populateScfToRVSDGPatterns(
    TypeConverter converter,
    RewritePatternSet &patterns);

/// Constructs the lower-funnel-shift pass.
std::unique_ptr<Pass> createQuantumOptimisePass();

/// Pass that legalizes multi-qubit quantum programs
/// such that they can be lowered to QIR
std::unique_ptr<Pass> createMultiQubitLegalizationPass();

std::unique_ptr<Pass> createScfToRVSDGPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace quantum

} // namespace mlir
