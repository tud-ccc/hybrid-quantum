/// Implements the multi-qubit legalization for Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

#include <complex>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <ostream>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_MULTIQUBITLEGALIZATION
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct MultiQubitLegalizationPass
        : mlir::quantum::impl::MultiQubitLegalizationBase<
              MultiQubitLegalizationPass> {
    using MultiQubitLegalizationBase::MultiQubitLegalizationBase;

    void runOnOperation() override;
};

struct TransformMultiQubitAllocation
        : public OneToNOpConversionPattern<AllocOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        OpAdaptor adaptor,
        OneToNPatternRewriter &rewriter) const override
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
        rewriter.replaceOp(op, qubits, adaptor.getResultMapping());
        return success();
    }
};

struct TransformDeallocate : public OneToNOpConversionPattern<DeallocateOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;

    LogicalResult matchAndRewrite(
        DeallocateOp op,
        OpAdaptor adaptor,
        OneToNPatternRewriter &rewriter) const override
    {
        const int64_t size = op.getInput().getType().getSize();
        if (size == 1)
            return rewriter.notifyMatchFailure(
                op,
                "Only multi-qubit allocations are transformed");

        auto loc = op.getLoc();
        for (int64_t i = 0; i < size; ++i) {
            auto qubit = adaptor.getOperands().front()[i];
            rewriter.create<DeallocateOp>(loc, qubit);
        }
        rewriter.replaceOp(op, ValueRange(), adaptor.getResultMapping());
        return success();
    }
};

struct TransformHOp : public OneToNOpConversionPattern<HOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;

    LogicalResult matchAndRewrite(
        HOp op,
        OpAdaptor adaptor,
        OneToNPatternRewriter &rewriter) const override
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
        rewriter.replaceOp(op, qubits, adaptor.getResultMapping());
        return success();
    }
};

struct TransformSplitOp : public OneToNOpConversionPattern<SplitOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;

    LogicalResult matchAndRewrite(
        SplitOp op,
        OpAdaptor adaptor,
        OneToNPatternRewriter &rewriter) const override
    {
        // auto loc = op.getLoc();
        //  llvm::SmallVector<Type> resultType{QubitType::get(getContext(), 1)};
        //  llvm::SmallVector<Value> qubits;
        //  for (int64_t i = 0; i < size; ++i) {
        //      auto inQubit = adaptor.getOperands().front()[i];
        //      auto genQubit = rewriter.create<SplitOp>(loc, resultType,
        //      inQubit); auto results = genQubit.getResults();
        //      qubits.append(results.begin(), results.end());
        //  }
        //  rewriter.replaceOp(op, qubits, adaptor.getResultMapping());
        auto qubits = adaptor.getOperands().front();
        rewriter.replaceOp(op, qubits, adaptor.getResultMapping());
        return success();
    }
};

struct TransformMergeOp : public OneToNOpConversionPattern<MergeOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;

    LogicalResult matchAndRewrite(
        MergeOp op,
        OpAdaptor adaptor,
        OneToNPatternRewriter &rewriter) const override
    {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];
        SmallVector<Value> vals;
        vals.append(lhs.begin(), lhs.end());
        vals.append(rhs.begin(), rhs.end());
        rewriter.replaceOp(op, vals, adaptor.getResultMapping());
        return success();
    }
};

} // namespace

void MultiQubitLegalizationPass::runOnOperation()
{
    auto context = &getContext();
    OneToNTypeConverter converter;
    RewritePatternSet patterns(context);

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

    populateMultiQubitLegalizationPatterns(converter, patterns);
    populateFuncTypeConversionPatterns(converter, patterns);

    if (failed(applyPartialOneToNConversion(
            getOperation(),
            converter,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateMultiQubitLegalizationPatterns(
    OneToNTypeConverter converter,
    RewritePatternSet &patterns)
{
    patterns.add<
        TransformMultiQubitAllocation,
        TransformDeallocate,
        TransformHOp,
        TransformSplitOp,
        TransformMergeOp>(converter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createMultiQubitLegalizationPass()
{
    return std::make_unique<MultiQubitLegalizationPass>();
}