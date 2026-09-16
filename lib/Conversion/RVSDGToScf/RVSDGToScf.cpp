/// Implements the RVSDG transformation to SCF dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/RVSDGToScf/RVSDGToScf.h"

#include "mlir/Pass/Pass.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

#include <algorithm>
#include <iostream>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/TypeRange.h>

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTRVSDGTOSCF
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

bool opResultsEqualOpArgs(rvsdg::GammaNode op)
{
    bool result = true;
    for (auto &region : op.getRegions()) {
        auto block = &region.front();
        auto yieldOp = block->getTerminator();

        result &= llvm::all_of_zip(
            block->getArguments(),
            yieldOp->getOperands(),
            [](Value arg, Value res) { return arg == res; });
    }
    return result;
}

struct ConvertRVSDGToScfPass
        : mlir::impl::ConvertRVSDGToScfBase<ConvertRVSDGToScfPass> {
    using ConvertRVSDGToScfBase::ConvertRVSDGToScfBase;

    void runOnOperation() override;
};

struct ConvertMatch : public OpConversionPattern<rvsdg::MatchOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::MatchOp op,
        rvsdg::MatchOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Support more than boolean 0 -> 1, 1 -> 0 mapping
        if (adaptor.getInput().getType() != rewriter.getI1Type())
            return failure();

        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

struct ConvertGamma : public OpConversionPattern<rvsdg::GammaNode> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::GammaNode op,
        rvsdg::GammaNodeAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO: We can have more than 2 regions (switch statement)
        if (op->getNumRegions() != 2) return failure();

        long elseSize = std::distance(
            op.getRegion(1).getOps().begin(),
            op.getRegion(1).getOps().end());

        if (opResultsEqualOpArgs(op) && elseSize == 1) {
            auto rvsdgYield = op->getRegion(0).front().getTerminator();
            rewriter.eraseOp(rvsdgYield);

            //  We only return captured arguments and do not have an else branch
            //  Create a simple if without results
            auto newIf = scf::IfOp::create(
                rewriter,
                op->getLoc(),
                adaptor.getPredicate(),
                /* withElseRegion */ false);

            Block* oldBlock = &op->getRegion(0).front();
            Block* newBlock = &newIf.getThenRegion().front();
            rewriter.inlineBlockBefore(
                oldBlock,
                newBlock,
                newBlock->begin(),
                adaptor.getInputs());

            rewriter.replaceOp(op, adaptor.getInputs());
        } else {
            auto newIf = scf::IfOp::create(
                rewriter,
                op->getLoc(),
                op->getResultTypes(),
                adaptor.getPredicate(),
                /* addThenBlock */ true,
                /* addElseBlock */ true);

            for (auto &&[oldRegion, newRegion] :
                 llvm::zip(op->getRegions(), newIf->getRegions())) {
                Block* oldBlock = &oldRegion.front();
                Block* newBlock = &newRegion.front();
                rewriter.inlineBlockBefore(
                    oldBlock,
                    newBlock,
                    newBlock->begin(),
                    adaptor.getInputs());
            }
            rewriter.replaceOp(op, newIf);
        }
        return success();
    }
};

struct ConvertYield : public OpConversionPattern<rvsdg::YieldOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::YieldOp op,
        rvsdg::YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
        return success();
    }
};

} // namespace

void ConvertRVSDGToScfPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });

    converter.addConversion([](rvsdg::ControlType type) -> std::optional<Type> {
        if (type.getNumOptions() != 2) return std::nullopt;

        return IntegerType::get(type.getContext(), 1);
    });

    target.addIllegalDialect<rvsdg::RVSDGDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<qillr::QILLRDialect>();

    populateConvertRVSDGToScfPatterns(converter, patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::rvsdg::populateConvertRVSDGToScfPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertGamma, ConvertYield, ConvertMatch>(
        typeConverter,
        patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertRVSDGToScfPass()
{
    return std::make_unique<ConvertRVSDGToScfPass>();
}
