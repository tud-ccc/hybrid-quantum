/// Implements the QILLR U Gate decomposition
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h"

#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::qillr;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qillr {

#define GEN_PASS_DEF_DECOMPOSEUGATES
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h.inc"

} // namespace mlir::qillr

//===----------------------------------------------------------------------===//

namespace {

struct DecomposeUGatesPass
        : mlir::qillr::impl::DecomposeUGatesBase<DecomposeUGatesPass> {
    using DecomposeUGatesBase::DecomposeUGatesBase;

    void runOnOperation() override;
};

struct DecomposeU3Pattern : public OpConversionPattern<qillr::U3Op> {
    using OpConversionPattern<qillr::U3Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::U3Op op,
        qillr::U3OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto qubit = adaptor.getInput();
        auto theta = adaptor.getTheta();
        auto phi = adaptor.getPhi();
        auto lambda = adaptor.getLambda();

        // U3(theta, phi, lambda) = Rz(phi) -> Ry(theta) -> Rz(lambda)
        rewriter.create<qillr::RzOp>(loc, qubit, phi);
        rewriter.create<qillr::RyOp>(loc, qubit, theta);
        rewriter.create<qillr::RzOp>(loc, qubit, lambda);

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

    target.addLegalOp<qillr::RzOp>();
    target.addLegalOp<qillr::RyOp>();
    target.addIllegalOp<qillr::U3Op>();

    // Fully qualify the function call
    populateUGatesDecompositionPatterns(patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::qillr::populateUGatesDecompositionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<DecomposeU3Pattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::qillr::createDecomposeUGatesPass()
{
    return std::make_unique<DecomposeUGatesPass>();
}
