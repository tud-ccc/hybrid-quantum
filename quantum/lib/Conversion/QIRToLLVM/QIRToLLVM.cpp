#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cinm-mlir/Dialect/QIR/IR/QIRDialect.h"
#include "cinm-mlir/Conversion/QIRPasses.h"
#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

namespace mlir::qir {

#define GEN_PASS_DEF_CONVERTQIRTOLLVMPASS
#include "cinm-mlir/Conversion/QIRPasses.h.inc"

struct QIRTypeConverter : public LLVMTypeConverter {
    QIRTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
        addConversion([&](qubitType type) { return convertQubitType(type); });
        addConversion([&](resultType type) { return convertResultType(type); });
    }

private:
    Type convertQubitType(Type mlirType) {
            return LLVM::LLVMStructType::getOpaque("Qubit", &getContext());
        }
    

    Type convertResultType(Type mlirType) {
        return LLVM::LLVMStructType::getOpaque("Result", &getContext());
    }
};


struct ConvertQIRToLLVMPass
    : public impl::ConvertQIRToLLVMPassBase<ConvertQIRToLLVMPass> {
    void runOnOperation() final {
        MLIRContext *context = &getContext();
        QIRTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        populateQIRToLLVMConversionPatterns(typeConverter, patterns);
        
        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        target.addIllegalDialect<QIRDialect>();

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};


std::unique_ptr<Pass> createConvertQIRToLLVMPass() {
    return std::make_unique<qir::ConvertQIRToLLVMPass>();
}

} // namespace mlir
