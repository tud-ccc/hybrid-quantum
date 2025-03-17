/// Implements the ConvertQuantumToQIRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "cinm-mlir/Dialect/QIR/IR/QIROps.h"
#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.h"
#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIR
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertQuantumToQIRPass
        : mlir::impl::ConvertQuantumToQIRBase<ConvertQuantumToQIRPass> {
    using ConvertQuantumToQIRBase::ConvertQuantumToQIRBase;

    void runOnOperation() override;
};

struct ConvertAlloc : public OpConversionPattern<quantum::AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opType = op.getResult().getType();
        if (!opType.isSingleQubit())
            return rewriter.notifyMatchFailure(
                op,
                "Please run --quantum-multi-qubit-legalize to transform "
                "multi-qubit into single-qubits.");

        rewriter.replaceOpWithNewOp<qir::AllocOp>(
            op,
            qir::QubitType::get(getContext()));
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertMeasure : public OpConversionPattern<quantum::MeasureOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();

        // Create new result type holding measurement value
        auto resultAlloc = rewriter.create<qir::AllocResultOp>(
            loc,
            qir::ResultType::get(op.getContext()));

        // Create new measure op with memory semantics; read from qubit
        // reference, store to result reference
        rewriter.create<qir::MeasureOp>(loc, adaptor.getInput(), resultAlloc);

        auto i1Type = rewriter.getI1Type();
        auto tensorType = mlir::RankedTensorType::get({1}, i1Type);

        // Read measurement in computational basis from result reference
        auto readMeasurement = rewriter.create<qir::ReadMeasurementOp>(
            loc,
            tensorType,
            resultAlloc.getResult());

        ValueRange replacements = {
            readMeasurement.getResult(),
            adaptor.getInput()};
        rewriter.replaceOp(op, replacements);
        return success();
    }
}; // struct ConvertMeasure

struct ConvertDealloc : public OpConversionPattern<quantum::DeallocateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        DeallocateOp op,
        DeallocateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<qir::ResetOp>(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertDealloc

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ftype = op.getFunctionType();

        auto genFuncTy = typeConverter->convertType(ftype);
        auto genFunc = rewriter.create<func::FuncOp>(
            op->getLoc(),
            op.getSymName(),
            llvm::dyn_cast<FunctionType>(genFuncTy));

        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFunc.getBody(),
                genFunc.end());
        }
        rewriter.replaceOp(op, genFunc);

        return success();
    }
}; // struct ConvertFunc

struct ConvertH : public OpConversionPattern<quantum::HOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        HOp op,
        HOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.create<qir::HOp>(op.getLoc(), adaptor.getInput());
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertSwap : public QuantumToQIROpConversion<quantum::SWAPOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        SWAPOp op,
        SWAPOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Retrieve the two input qubits from the adaptor.
        Value qubit1 = adaptor.getLhs();
        Value qubit2 = adaptor.getRhs();
        auto qirQubit1 = mapping->find(qubit1)[0];
        auto qirQubit2 = mapping->find(qubit2)[0];
        mapping->allocate(op.getResult1(), qirQubit1);
        mapping->allocate(op.getResult2(), qirQubit2);
        rewriter.create<qir::SwapOp>(op.getLoc(), qirQubit1, qirQubit2);
        rewriter.replaceOp(op, {qubit1, qubit2});
        return success();
    }
};
} // namespace

void ConvertQuantumToQIRPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](quantum::QubitType ty) {
        return qir::QubitType::get(ty.getContext());
    });
    typeConverter.addConversion([&](FunctionType fty) {
        llvm::SmallVector<Type> argTypes, resTypes;

        for (auto ins : fty.getInputs())
            argTypes.push_back(typeConverter.convertType(ins));

        for (auto res : fty.getResults())
            resTypes.push_back(typeConverter.convertType(res));

        return FunctionType::get(fty.getContext(), argTypes, resTypes);
    });

    quantum::populateConvertQuantumToQIRPatterns(typeConverter, patterns);

    target.addIllegalDialect<quantum::QuantumDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addLegalDialect<qir::QIRDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isLegal(op.getFunctionType());
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        return signalPassFailure();
}

void mlir::quantum::populateConvertQuantumToQIRPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc,
        ConvertMeasure,
        ConvertH,
        ConvertFunc,
        ConvertSwap,
        ConvertDealloc>(typeConverter, patterns.getContext(), /* benefit*/ 1);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQIRPass()
{
    return std::make_unique<ConvertQuantumToQIRPass>();
}
