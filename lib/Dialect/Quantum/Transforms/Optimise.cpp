/// Implements the Optimise.
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
#define GEN_PASS_DEF_HERMITIANCANCEL
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum
//===----------------------------------------------------------------------===//

namespace {
/// Pattern: Drop phase gates immediately before measurement
struct DropPhaseBeforeMeasure : OpRewritePattern<quantum::ZOp> {
    using OpRewritePattern<quantum::ZOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(quantum::ZOp zOp, PatternRewriter &rewriter) const override
    {
        Value zResult = zOp.getResult();
        if (!zResult.getType().isa<quantum::QubitType>()) return failure();
        if (zResult.use_empty()) return failure();

        for (Operation* userOp : zResult.getUsers()) {
            auto measureOp = dyn_cast<quantum::MeasureOp>(userOp);
            if (!measureOp) return failure();
            // Ensure that the qubit operand of the MeasureOp is exactly zResult
            if (measureOp.getOperand() != zResult) return failure();
        }
        rewriter.replaceOp(zOp, zOp.getInput());
        return success();
    }
};

/// Pattern: Cancel double Hermitian ops
struct FoldDoubleHermitian : OpTraitRewritePattern<Hermitian> {
    using OpTraitRewritePattern::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override
    {
        auto inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName()) return failure();

        rewriter.replaceOp(op, inner->getOperand(0));
        return success();
    }
};
} // namespace

/// --quantum-optimise
struct QuantumOptimisePass
        : quantum::impl::QuantumOptimiseBase<QuantumOptimisePass> {
    using QuantumOptimiseBase::QuantumOptimiseBase;

    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<DropPhaseBeforeMeasure>(patterns.getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

/// --hermitian-peephole
struct HermitianCancelPass
        : quantum::impl::HermitianCancelBase<HermitianCancelPass> {
    using HermitianCancelBase::HermitianCancelBase;

    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<FoldDoubleHermitian>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}

std::unique_ptr<Pass> mlir::quantum::createHermitianCancelPass()
{
    return std::make_unique<HermitianCancelPass>();
}
