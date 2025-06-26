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
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

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
        auto userOp = *op.getResult().getUsers().begin();
        if (dyn_cast<MeasureOp>(userOp) || dyn_cast<MeasureSingleOp>(userOp)) {
            rewriter.replaceOp(op, op.getOperand());
            return success();
        }
        return failure();
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
    patterns.add<
        DropPhaseBeforeMeasure<ZOp>,
        DropPhaseBeforeMeasure<SOp>,
        DropPhaseBeforeMeasure<TOp>,
        DropPhaseBeforeMeasure<SdgOp>,
        DropPhaseBeforeMeasure<TdgOp>>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}
