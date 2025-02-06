/// Implements the ConvertQIRToLLVMPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOLLVM
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//


namespace {

struct ConvertQIRToLLVMPass
        : mlir::impl::ConvertQIRToLLVMBase<ConvertQIRToLLVMPass> {
    using ConvertQIRToLLVMBase::ConvertQIRToLLVMBase;

    void runOnOperation() override;
};

struct ConvertMeasureOp : ConvertOpToLLVMPattern<MeasureOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        return failure();
    }
}; // struct ConvertMeasureOp

} // namespace

void ConvertQIRToLLVMPass::runOnOperation()
{
    LLVMTypeConverter typeConverter(&getContext());
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    qir::populateConvertQIRToLLVMPatterns(typeConverter, patterns);
    //mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        
    target.addIllegalDialect<qir::QIRDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

void mlir::qir::populateConvertQIRToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertMeasureOp>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}





