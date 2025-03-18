/// Implements the Optimise.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_QUANTUMOPTIMISE
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct QuantumOptimisePass
        : mlir::quantum::impl::QuantumOptimiseBase<QuantumOptimisePass> {
    using QuantumOptimiseBase::QuantumOptimiseBase;

    void runOnOperation() override;
};

struct CancelHadamardPairs : public OpRewritePattern<HOp> {
    using OpRewritePattern<HOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(HOp op, PatternRewriter &rewriter) const override
    {
        Operation* nextOp = op->getNextNode();
        if (nextOp && isa<HOp>(nextOp)) {
            HOp nextHadamard = cast<HOp>(nextOp);
            // Check that the result of the first H is the input to the next H.
            if (op.getResult() == nextHadamard.getInput()) {
                // Replace the two H operations with the original input.
                rewriter.replaceOp(nextHadamard, op.getInput());
                rewriter.eraseOp(op);
                return success();
            }
        }
        return failure();
    }
};

template<typename PauliOp>
struct CancelPauliPairs : public OpRewritePattern<PauliOp> {
    using OpRewritePattern<PauliOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(PauliOp op, PatternRewriter &rewriter) const override
    {
        Operation* nextOp = op->getNextNode();
        if (!nextOp || !isa<PauliOp>(nextOp))
            return failure(); // Ensure the next operation is the same type

        PauliOp nextPauli = cast<PauliOp>(nextOp);

        // Ensure the result of the first Op is the input to the next Op
        if (op.getResult() != nextPauli.getOperand())
            return failure(); // They must act on the same qubit

        // Replace the second Pauli Op with the original qubit input
        rewriter.replaceOp(nextPauli, op.getOperand());
        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

void QuantumOptimisePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    populateQuantumOptimisePatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateQuantumOptimisePatterns(RewritePatternSet &patterns)
{
    patterns.add<CancelHadamardPairs>(patterns.getContext());
    patterns.add<CancelPauliPairs<XOp>>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}