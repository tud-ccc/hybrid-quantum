/// Implements the QIR U Gate decomposition
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"
#include "quantum-mlir/Dialect/QIR/Transforms/Passes.h"

#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qir {

#define GEN_PASS_DEF_DECOMPOSEUGATES
#include "quantum-mlir/Dialect/QIR/Transforms/Passes.h.inc"

} // namespace mlir::qir

//===----------------------------------------------------------------------===//

namespace {

struct DecomposeUGatesPass
        : mlir::qir::impl::DecomposeUGatesBase<DecomposeUGatesPass> {
    using DecomposeUGatesBase::DecomposeUGatesBase;

    void runOnOperation() override;
};

struct DecomposeU3Pattern : public OpConversionPattern<qir::U3Op> {
    using OpConversionPattern<qir::U3Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::U3Op op,
        qir::U3OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto qubit = adaptor.getInput();
        auto theta = adaptor.getTheta();
        auto phi = adaptor.getPhi();
        auto lambda = adaptor.getLambda();

        // U3(theta, phi, lambda) = Rz(phi) -> Ry(theta) -> Rz(lambda)
        rewriter.create<qir::RzOp>(loc, qubit, phi);
        rewriter.create<qir::RyOp>(loc, qubit, theta);
        rewriter.create<qir::RzOp>(loc, qubit, lambda);

        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

void DecomposeUGatesPass::runOnOperation()
{
    auto context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalOp<qir::RzOp>();
    target.addLegalOp<qir::RyOp>();
    target.addIllegalOp<qir::U3Op>();

    // Fully qualify the function call
    populateUGatesDecompositionPatterns(patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::qir::populateUGatesDecompositionPatterns(RewritePatternSet &patterns)
{
    patterns.add<DecomposeU3Pattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::qir::createDecomposeUGatesPass()
{
    return std::make_unique<DecomposeUGatesPass>();
}
