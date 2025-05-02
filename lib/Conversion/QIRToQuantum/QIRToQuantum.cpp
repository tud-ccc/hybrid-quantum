/// Implements the ConvertQuantumToQIRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/QIRToQuantum/QIRToQuantum.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRBase.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRTypes.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

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
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOQUANTUM
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertQIRToQuantumPass
        : mlir::impl::ConvertQIRToQuantumBase<ConvertQIRToQuantumPass> {
    using ConvertQIRToQuantumBase::ConvertQIRToQuantumBase;

    void runOnOperation() override;
};

struct ConvertAlloc : public OpConversionPattern<qir::AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<quantum::AllocOp>(
            op,
            quantum::QubitType::get(getContext(), 1));
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertResultAlloc : public OpConversionPattern<qir::AllocResultOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocResultOp op,
        AllocResultOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // We do not have a representation for result registers in Quantum
        // dialect
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertResultAllocOp

struct ConvertSwap : public OpConversionPattern<qir::SwapOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::SwapOp op,
        qir::SwapOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Retrieve the two input qubits from the adaptor.
        Value qubit1 = adaptor.getLhs();
        Value qubit2 = adaptor.getRhs();
        rewriter.create<quantum::SWAPOp>(op.getLoc(), qubit1, qubit2);
        rewriter.replaceOp(op, {qubit1, qubit2});
        return success();
    }
};
} // namespace

struct ConvertH : public OpConversionPattern<qir::HOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        HOp op,
        HOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.create<quantum::HOp>(op.getLoc(), adaptor.getInput());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertHOp

struct ConvertRz : public OpConversionPattern<qir::RzOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        RzOp op,
        RzOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.create<quantum::RzOp>(
            op.getLoc(),
            adaptor.getInput(),
            adaptor.getAngle());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertHOp

void ConvertQIRToQuantumPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](qir::QubitType ty) {
        return quantum::QubitType::get(ty.getContext(), 1);
    });
    typeConverter.addConversion([&](FunctionType fty) {
        llvm::SmallVector<Type> argTypes, resTypes;

        for (auto ins : fty.getInputs())
            argTypes.push_back(typeConverter.convertType(ins));

        for (auto res : fty.getResults())
            resTypes.push_back(typeConverter.convertType(res));

        return FunctionType::get(fty.getContext(), argTypes, resTypes);
    });

    qir::populateConvertQIRToQuantumPatterns(typeConverter, patterns);

    target.addIllegalDialect<qir::QIRDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addLegalDialect<quantum::QuantumDialect>();
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

void mlir::qir::populateConvertQIRToQuantumPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc,
        ConvertSwap,
        ConvertResultAlloc,
        ConvertH,
        ConvertRz>(
        typeConverter,
        patterns.getContext(),
        /* benefit*/ 1);
}

std::unique_ptr<Pass> mlir::createConvertQIRToQuantumPass()
{
    return std::make_unique<ConvertQIRToQuantumPass>();
}
