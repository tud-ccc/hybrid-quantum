/// Implements the RVSDG transformation for RVSDG dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/RVSDGConversion/RVSDGConversion.h"

#include "mlir/Pass/Pass.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTRVSDG
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertRVSDGPass : mlir::impl::ConvertRVSDGBase<ConvertRVSDGPass> {
    using ConvertRVSDGBase::ConvertRVSDGBase;

    void runOnOperation() override;
};

struct ConvertRVSDGGamma : public OpConversionPattern<rvsdg::GammaNode> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::GammaNode op,
        rvsdg::GammaNodeAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto converter = getTypeConverter();

        SmallVector<Type> convertedResultTypes;
        if (failed(converter->convertTypes(
                op->getResultTypes(),
                convertedResultTypes)))
            return failure();

        auto newGamma = rvsdg::GammaNode::create(
            rewriter,
            op->getLoc(),
            convertedResultTypes,
            adaptor.getPredicate(),
            adaptor.getInputs(),
            op->getNumRegions());

        for (auto &&[oldRegion, newRegion] :
             llvm::zip(op->getRegions(), newGamma->getRegions())) {
            rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
        }

        // Change the block argument types of each region
        for (Region &r : newGamma->getRegions()) {
            auto newRegion = rewriter.convertRegionTypes(&r, *converter);
            if (failed(newRegion)) return failure();
        }

        rewriter.replaceOp(op, newGamma);
        return success();
    }
};

struct ConvertRVSDGYield : public OpConversionPattern<rvsdg::YieldOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::YieldOp op,
        rvsdg::YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<rvsdg::YieldOp>(op, adaptor.getOperands());
        return success();
    }
};

} // namespace

void ConvertRVSDGPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });

    target.addLegalDialect<rvsdg::RVSDGDialect>();

    populateConvertRVSDGPatterns(converter, patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::rvsdg::populateConvertRVSDGPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertRVSDGGamma, ConvertRVSDGYield>(
        typeConverter,
        patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertRVSDGPass()
{ return std::make_unique<ConvertRVSDGPass>(); }
