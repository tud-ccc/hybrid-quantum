/// Implements the Optimise.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_QUANTUMOPTIMISE
#define GEN_PASS_DEF_HERMITIANCANCEL
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum
//===----------------------------------------------------------------------===//

// Collection of patterns to optimize quantum operations
// USAGE: quantum-opt --quantum-optimise <input.mlir>
struct QuantumOptimisePass
        : mlir::quantum::impl::QuantumOptimiseBase<QuantumOptimisePass> {
    using QuantumOptimiseBase::QuantumOptimiseBase;

    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        populateQuantumOptimisePatterns(patterns); // Add patterns to the set
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    };
};

// Patterns to cancel hermitian operations
// USAGE:  quantum-opt --hermitian-peephole <input.mlir>
struct HermitianCancelPass
        : mlir::quantum::impl::HermitianCancelBase<HermitianCancelPass> {
    using HermitianCancelBase::HermitianCancelBase;

    void runOnOperation() override
    {
        struct FoldDoubleHermitian : OpTraitRewritePattern<Hermitian> {
            using OpTraitRewritePattern::OpTraitRewritePattern;

            LogicalResult matchAndRewrite(
                Operation* op,
                PatternRewriter &rewriter) const override
            {
                auto inner = op->getOperand(0).getDefiningOp();
                if (!inner || inner->getName() != op->getName())
                    return failure();

                rewriter.replaceOp(op, inner->getOperand(0));
                return success();
            }
        };

        RewritePatternSet patterns(&getContext());
        patterns.add<FoldDoubleHermitian>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

void mlir::quantum::populateQuantumOptimisePatterns(RewritePatternSet &patterns)
{
    // TBD: Add more generic patterns here
    // USAGE: quantum-opt --quantum-optimise <input.mlir>
}

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}

std::unique_ptr<Pass> mlir::quantum::createHermitianCancelPass()
{
    return std::make_unique<HermitianCancelPass>();
}
