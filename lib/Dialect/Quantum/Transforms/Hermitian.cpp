/// Implements the Herimitan trait cancelling.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washimneupane@outlook.com)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_HERMITIANCANCEL
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct HermitianCancelPass
        : quantum::impl::HermitianCancelBase<HermitianCancelPass> {
    using HermitianCancelBase::HermitianCancelBase;

    void runOnOperation() override;
};

/// Pattern: Cancel double Hermitian ops
struct FoldDoubleHermitian : OpTraitRewritePattern<Hermitian> {
    using OpTraitRewritePattern::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override
    {
        bool isSame = std::all_of(
            op->getOperands().begin(),
            op->getOperands().end(),
            [&](auto operand) {
                auto inner = operand.getDefiningOp();
                return inner && inner->template hasTrait<Hermitian>()
                       && inner->getName() == op->getName();
            });

        if (isSame) {
            auto inner = op->getOperand(0).getDefiningOp();
            rewriter.replaceOp(op, inner->getOperands());
            return success();
        }
        return failure();
    }
};

} // namespace

void HermitianCancelPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    populateHermitianCancelPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateHermitianCancelPatterns(RewritePatternSet &patterns)
{
    patterns.add<FoldDoubleHermitian>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createHermitianCancelPass()
{
    return std::make_unique<HermitianCancelPass>();
}
