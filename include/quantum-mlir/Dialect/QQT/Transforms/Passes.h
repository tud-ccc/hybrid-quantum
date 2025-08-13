/// Declares the QQT passes.
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qqt {

/// Constructs the pass that moves load and store
/// instructions to their dominating predecessor
/// and post-dominating successor.
std::unique_ptr<Pass> createLoadStoreMovePass();

std::unique_ptr<Pass> createLoadStoreEliminationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h.inc"

} // namespace qqt

} // namespace mlir
