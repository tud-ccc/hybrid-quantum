#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"
#include "cinm-mlir/Conversion/QuantumPasses.h"
#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

namespace mlir::quantum {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIRPASS
#include "cinm-mlir/Conversion/QuantumPasses.h.inc"

// struct QIRTypeConverter : public TypeConverter {
//     QIRTypeConverter(MLIRContext *ctx) : TypeConverter() {
//         addConversion([&](qubitType type) { return convertQubitType(type); });
//         //addConversion([&](resultType type) { return convertResultType(type); });
//     }

// private:
//     Type convertQubitType(Type mlirType) {
//             //return LLVM::LLVMStructType::getOpaque("Qubit", &getContext());
//     }
    

//     //Type convertResultType(Type mlirType) {
//     //    return LLVM::LLVMStructType::getOpaque("Result", &getContext());
//     //}
// };


struct ConvertQuantumToQIRPass
    : public impl::ConvertQuantumToQIRPassBase<ConvertQuantumToQIRPass> {
    void runOnOperation() final {
        MLIRContext *context = &getContext();
        //QIRTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        //populateQuantumToQIRConversionPatterns(typeConverter, patterns);
        
        ConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        target.addIllegalDialect<QuantumDialect>();

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};


std::unique_ptr<Pass> createConvertQuantumToQIRPass() {
    return std::make_unique<quantum::ConvertQuantumToQIRPass>();
}

} // namespace mlir
