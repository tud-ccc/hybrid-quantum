/// Implements the gate optimization.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washimneupane@outlook.com)

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
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct QuantumOptimisePass
        : mlir::quantum::impl::QuantumOptimiseBase<QuantumOptimisePass> {
    using QuantumOptimiseBase::QuantumOptimiseBase;

    void runOnOperation() override;
};

/// Pattern: Drop phase gates immediately before measurement
template<typename PhaseOp>
struct DropPhaseBeforeMeasure : OpRewritePattern<PhaseOp> {
    using OpRewritePattern<PhaseOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(PhaseOp op, PatternRewriter &rewriter) const override
    {
        // Check if op has 1 use and that use is a MeasureOp
        //  If not, do nothing. Else, replace the op with its operand (the
        //  only user).

        if (!op.getResult().hasOneUse()) return failure();
        auto* user = *op.getResult().user_begin();
        if (!isa<quantum::MeasureOp>(user)) return failure();
        rewriter.replaceOp(op, op.getOperand());
        return success();
    }
};

} // namespace

void QuantumOptimisePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    populateQuantumOptimisePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateQuantumOptimisePatterns(RewritePatternSet &patterns)
{
    patterns.add<DropPhaseBeforeMeasure<ZOp>>(patterns.getContext());
    patterns.add<DropPhaseBeforeMeasure<SOp>>(patterns.getContext());
    patterns.add<DropPhaseBeforeMeasure<TOp>>(patterns.getContext());
    patterns.add<DropPhaseBeforeMeasure<SdgOp>>(patterns.getContext());
    patterns.add<DropPhaseBeforeMeasure<TdgOp>>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}
