/// Implements the multi-qubit legalization for Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_MULTIQUBITLEGALIZATION
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct MultiQubitLegalizationPass
        : mlir::quantum::impl::MultiQubitLegalizationBase<
              MultiQubitLegalizationPass> {
    using MultiQubitLegalizationBase::MultiQubitLegalizationBase;

    void runOnOperation() override;
};

struct TransformMultiQubitAllocation : public OpConversionPattern<AllocOp> {
    using OpConversionPattern<AllocOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const int64_t size = op.getType().getSize();
        if (size == 1)
            return rewriter.notifyMatchFailure(
                op,
                "Only multi-qubit allocations are transformed");

        auto loc = op.getLoc();
        llvm::SmallVector<Value> qubits;
        for (int64_t i = 0; i < size; ++i) {
            auto alloc =
                rewriter.create<AllocOp>(loc, QubitType::get(getContext(), 1));
            qubits.push_back(alloc.getResult());
        }
        rewriter.replaceOpWithMultiple(op, {qubits});
        return success();
    }
};

struct TransformDeallocate : public OpConversionPattern<DeallocateOp> {
    using OpConversionPattern<DeallocateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        DeallocateOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const int64_t size = op.getInput().getType().getSize();
        if (size == 1)
            return rewriter.notifyMatchFailure(
                op,
                "Only multi-qubit allocations are transformed");

        auto loc = op.getLoc();
        for (int64_t i = 0; i < size; ++i) {
            auto qubit = adaptor.getInput()[i];

            rewriter.create<DeallocateOp>(loc, qubit);
        }

        rewriter.replaceOpWithMultiple(op, {});
        return success();
    }
};

struct TransformHOp : public OpConversionPattern<HOp> {
    using OpConversionPattern<HOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        HOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const int64_t size = op.getInput().getType().getSize();
        if (size == 1)
            return rewriter.notifyMatchFailure(
                op,
                "Only multi-qubit allocations are transformed");

        auto loc = op.getLoc();
        llvm::SmallVector<Value> qubits;
        for (int64_t i = 0; i < size; ++i) {
            auto inQubit = adaptor.getOperands().front()[i];
            auto genQubit = rewriter.create<HOp>(
                loc,
                QubitType::get(getContext(), 1),
                inQubit);
            qubits.push_back(genQubit.getResult());
        }
        rewriter.replaceOpWithMultiple(op, {qubits});
        return success();
    }
};

struct TransformSplitOp : public OpConversionPattern<SplitOp> {
    using OpConversionPattern<SplitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        SplitOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<ValueRange> replacements;
        size_t offset = 0;
        for (int i = 0, e = op->getNumResults(); i < e; ++i) {
            auto val = op->getOpResults()[i];
            auto res =
                llvm::dyn_cast<mlir::TypedValue<quantum::QubitType>>(val);
            auto dim = res.getType().getSize();
            auto view = adaptor.getInput().slice(offset, dim);
            replacements.push_back(view);
            offset += dim;
        }

        rewriter.replaceOpWithMultiple(op, replacements);
        return success();
    }
};

struct TransformMergeOp : public OpConversionPattern<MergeOp> {
    using OpConversionPattern<MergeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MergeOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];
        SmallVector<Value> vals;
        vals.append(lhs.begin(), lhs.end());
        vals.append(rhs.begin(), rhs.end());
        rewriter.replaceOpWithMultiple(op, {vals});
        return success();
    }
};

struct TransformMeasureOp : public OpConversionPattern<MeasureOp> {
    using OpConversionPattern<MeasureOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        OneToNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const int64_t size = op.getInput().getType().getSize();
        if (size == 1)
            return rewriter.notifyMatchFailure(
                op,
                "Only multi-qubit allocations are transformed");

        auto loc = op.getLoc();
        auto i1Type = rewriter.getI1Type();
        auto genTensorType = mlir::RankedTensorType::get({1}, i1Type);

        llvm::SmallVector<Value> qubitResults;
        llvm::SmallVector<Value> measureResults;

        for (int64_t i = 0; i < size; ++i) {
            auto inQubit = adaptor.getInput()[i];
            auto measure = rewriter.create<MeasureOp>(
                loc,
                genTensorType,
                QubitType::get(getContext(), 1),
                inQubit);

            measureResults.push_back(measure.getMeasurement());
            qubitResults.push_back(measure.getResult());
        }
        // TODO: Use fromElements
        auto concatenatedTensor =
            rewriter.create<tensor::ConcatOp>(loc, 0, measureResults);

        SmallVector<ValueRange> replacements;
        replacements.push_back({concatenatedTensor.getResult()});
        replacements.push_back(qubitResults);
        rewriter.replaceOpWithMultiple(op, replacements);
        return success();
    }
};

} // namespace

void MultiQubitLegalizationPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [](QubitType type,
           llvm::SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
            // A qubit<1> does not need conversion
            if (type.isSingleQubit()) return std::nullopt;
            // Convert a qubit<N> to N x qubit<1>
            types = SmallVector<Type>(
                type.getSize(),
                QubitType::get(type.getContext(), 1));
            return success();
        });
    converter.addConversion(
        [](RankedTensorType type,
           llvm::SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
            // A qubit<1> does not need conversion
            if (type.getRank() == 1) return std::nullopt;
            // Convert a tensor<Nx?> to N times tensor<1x?>
            types = SmallVector<Type>(
                type.getRank(),
                mlir::RankedTensorType::get({1}, type.getElementType()));
            return success();
        });

    target.addLegalDialect<tensor::TensorDialect>();
    target.addDynamicallyLegalOp<quantum::AllocOp>(
        [&](quantum::AllocOp op) { return converter.isLegal(op.getType()); });
    target.addDynamicallyLegalOp<quantum::DeallocateOp>(
        [&](quantum::DeallocateOp op) {
            return converter.isLegal(op.getInput().getType());
        });
    target.addDynamicallyLegalOp<quantum::MeasureOp>(
        [&](quantum::MeasureOp op) {
            return converter.isLegal(op.getInput().getType());
        });
    target.addDynamicallyLegalOp<quantum::HOp>([&](quantum::HOp op) {
        return converter.isLegal(op.getInput().getType());
    });
    target.addDynamicallyLegalOp<quantum::SplitOp>(
        [](quantum::SplitOp op) { return false; });
    target.addDynamicallyLegalOp<quantum::MergeOp>(
        [](quantum::MergeOp op) { return false; });

    populateMultiQubitLegalizationPatterns(converter, patterns);
    populateFuncTypeConversionPatterns(converter, patterns);

    // applyPartialOneToNConversion
    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateMultiQubitLegalizationPatterns(
    TypeConverter converter,
    RewritePatternSet &patterns)
{
    patterns.add<
        TransformMultiQubitAllocation,
        TransformDeallocate,
        TransformHOp,
        TransformSplitOp,
        TransformMergeOp,
        TransformMeasureOp>(converter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createMultiQubitLegalizationPass()
{
    return std::make_unique<MultiQubitLegalizationPass>();
}