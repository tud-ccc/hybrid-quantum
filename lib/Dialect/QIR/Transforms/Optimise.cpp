#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"
#include "quantum-mlir/Dialect/QIR/Transforms/Passes.h" // <--- IMPORTANT

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated pass definitions
//===----------------------------------------------------------------------===//

namespace mlir::qir {
#define GEN_PASS_DEF_QIROPTIMISE
#include "quantum-mlir/Dialect/QIR/Transforms/Passes.h.inc"
} // namespace mlir::qir

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

namespace {
struct DecomposeU3Pattern : mlir::RewritePattern {
    DecomposeU3Pattern(mlir::MLIRContext* context)
            : RewritePattern("qir.U3", 1, context)
    {}

    LogicalResult
    matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override
    {
        auto u3 = llvm::dyn_cast<mlir::qir::U3Op>(op);
        if (!u3) return failure();

        Location loc = u3.getLoc();
        Value qubit = u3.getInput();
        Value theta = u3.getTheta();
        Value phi = u3.getPhi();
        Value lambda = u3.getLambda();

        // U3(θ, φ, λ) = Rz(φ) -> Ry(θ) -> Rz(λ)
        rewriter.create<qir::RzOp>(loc, qubit, phi);
        rewriter.create<qir::RyOp>(loc, qubit, theta);
        rewriter.create<qir::RzOp>(loc, qubit, lambda);
        rewriter.eraseOp(op); // ✅

        return success();
    }
};
} // namespace

struct QIROptimisePass : mlir::qir::impl::QIROptimiseBase<QIROptimisePass> {
    using QIROptimiseBase::QIROptimiseBase;

    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());

        // Fully qualify the function call
        mlir::qir::populateQIROptimisePatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(
                getOperation(),
                std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration + Pattern Population
//===----------------------------------------------------------------------===//

namespace mlir::qir {

void populateQIROptimisePatterns(RewritePatternSet &patterns)
{
    patterns.add<DecomposeU3Pattern>(patterns.getContext());
}

std::unique_ptr<Pass> createQIROptimisePass()
{
    return std::make_unique<QIROptimisePass>();
}

} // namespace mlir::qir
