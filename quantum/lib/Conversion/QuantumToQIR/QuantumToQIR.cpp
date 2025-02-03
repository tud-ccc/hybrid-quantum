#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cinm-mlir/Dialect/QIR/IR/QIRDialect.h"
#include "cinm-mlir/Dialect/QIR/IR/QIROps.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"
#include "cinm-mlir/Conversion/QuantumPasses.h"
#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

namespace mlir::quantum {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIRPASS
#include "cinm-mlir/Conversion/QuantumPasses.h.inc"


// class QuantumTypeConverter : public TypeConverter {
// public:
//     using TypeConverter::convertType;

//     QuantumTypeConverter(MLIRContext *ctx) : TypeConverter() {
//         //addConversion([](Type type) { return type; });
//         addConversion([&](quantum::QubitType type) {
//             return qir::QubitType::get(getContext());
//         });
// //         addConversion([&](qubitType type) { return convertQubitType(type); });
// //         //addConversion([&](resultType type) { return convertResultType(type); });
//     }

//     MLIRContext *getContext() const { return context; }
// private:
//     MLIRContext *context;
// };

struct QuantumToQIRTarget : public ConversionTarget {
  QuantumToQIRTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    //addLegalDialect<StandardOpsDialect>();
    addLegalDialect<qir::QIRDialect>();
    //addLegalDialect<AffineDialect>();

    addIllegalDialect<quantum::QuantumDialect>();
  }
};

struct ConvertQuantumToQIRPass
    : public impl::ConvertQuantumToQIRPassBase<ConvertQuantumToQIRPass> {
    void runOnOperation() final {
        TypeConverter typeConverter;
        RewritePatternSet patterns(&getContext());        
        populateQuantumToQIRConversionPatterns(typeConverter, patterns);

        QuantumToQIRTarget target(getContext());
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

//} // namespace mlir::quantum

//namespace mlir {

std::unique_ptr<Pass> createConvertQuantumToQIRPass() {
    return std::make_unique<quantum::ConvertQuantumToQIRPass>();
}

} // namespace mlir
