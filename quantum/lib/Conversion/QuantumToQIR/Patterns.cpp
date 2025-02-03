#include <string>
#include "mlir/Transforms/DialectConversion.h"

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

using namespace mlir;
using namespace mlir::quantum;

namespace {

struct AllocOpPattern : public OpConversionPattern<AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AllocOp op, AllocOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        // Extract the size from the result type (!quantum.qubit<size>)
        auto resultType = op.getResult().getType();
        unsigned size = resultType.getSize();

        // QIR function name and signature
        //StringRef qirName = "__quantum__rt__qubit_allocate_array";
        //Type qubitPtrTy = conv->convertType(resultType); // Convert !quantum.qubit<size> to LLVM type
        //Type qirSignature = LLVM::LLVMFunctionType::get(qubitPtrTy, {IntegerType::get(ctx, 64)});

        // Ensure function declaration exists
        //LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        //if (!fnDecl) {
        //    return rewriter.notifyMatchFailure(op, "Failed to create or find function declaration.");
        //}

        // Create constant for the size
        //Value nQubits = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(size));

        // Replace the operation with a call to QIR function
        //rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, qubitPtrTy, fnDecl.getSymName(), nQubits);

        //return success();
        return failure();
    }
}; // struct AllocOpPattern

struct MeasureOpPattern : public OpConversionPattern<MeasureOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(MeasureOp op, MeasureOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        return failure();
    }
}; // struct MeasureOpPattern

}

namespace mlir::quantum {
void populateQuantumToQIRConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    //patterns.add<AllocOpPattern>(typeConverter, patterns.getContext());
    patterns.add<MeasureOpPattern>(typeConverter, patterns.getContext());
}

} // namespace quantum
