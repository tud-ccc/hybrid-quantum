/// Implements the ConvertQuantumToQIRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/QIR/IR/QIR.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIR
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

struct mlir::quantum::QuantumToQirQubitTypeMapping {
public:
    QuantumToQirQubitTypeMapping() : map() {}

    // Map a `quantum.Qubit` to (possible many) `qir.qubit`
    void allocate(Value quantumQubit, ValueRange qirQubits) {
        for (auto qirQubit : qirQubits) {
            map[quantumQubit].push_back(qirQubit);
        }
    }

    llvm::SmallVector<Value> find(Value quantum) {
        return map[quantum];
    }

private:
    llvm::DenseMap<Value, llvm::SmallVector<Value>> map;
};

namespace {

struct ConvertQuantumToQIRPass
        : mlir::impl::ConvertQuantumToQIRBase<ConvertQuantumToQIRPass> {
    using ConvertQuantumToQIRBase::ConvertQuantumToQIRBase;

    void runOnOperation() override;
};

template <typename Op>
struct QuantumToQIROpConversion : OpConversionPattern<Op> {
    explicit QuantumToQIROpConversion(MLIRContext *context, QuantumToQirQubitTypeMapping *mapping)
        : OpConversionPattern<Op>(context, /* benefit */ 1), mapping(mapping) {}

    QuantumToQirQubitTypeMapping *mapping;
};

struct ConvertAlloc : public QuantumToQIROpConversion<quantum::AllocOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        unsigned size = op.getType().getSize();
        llvm::SmallVector<Value> qubits;
        for(unsigned i = 0; i < size; i++) {
            auto qubit = rewriter.create<qir::AllocOp>(
                op.getLoc(),
                qir::QubitType::get(getContext()));
            qubits.push_back(qubit);
        }
        mapping->allocate(op.getResult(), qubits);
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertMeasure : public QuantumToQIROpConversion<quantum::MeasureOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto inputQubit = op.getInput();
        auto resultQubit = op.getResult();
        auto measurementResult = op.getMeasurement();

        // Map resulting qubit to qubit memory reference
        auto qirInput = mapping->find(inputQubit)[0];
        mapping->allocate(resultQubit, qirInput);
        
        // Create new result type holding measurement value
        auto resultDef = rewriter.create<qir::AllocResultOp>(
            op.getLoc(),
            qir::ResultType::get(getContext())).getResult();

        auto measureOp = rewriter.create<qir::MeasureOp>(
            op.getLoc(),
            qirInput,
            resultDef).getResult();
            
        rewriter.eraseOp(op);

        // Replace direct uses of the measurement value with QIR values
        auto result = rewriter.create<qir::ReadMeasurementOp>(
            op.getLoc(),
            measurementResult.getType(),
            resultDef).getResult();

        measurementResult.replaceAllUsesWith(result);

        return success();
    }
}; // struct ConvertMeasure

} // namespace

void ConvertQuantumToQIRPass::runOnOperation()
{
  TypeConverter typeConverter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  QuantumToQirQubitTypeMapping mapping;

  quantum::populateConvertQuantumToQIRPatterns(typeConverter, mapping, patterns);

  target.addIllegalDialect<quantum::QuantumDialect>();
  target.addLegalDialect<qir::QIRDialect>();

  if (failed(applyPartialConversion(
          getOperation(),
          target,
          std::move(patterns)))) {
    return signalPassFailure();
  }
}

void mlir::quantum::populateConvertQuantumToQIRPatterns(
    TypeConverter &typeConverter,
    QuantumToQirQubitTypeMapping &mapping,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc,
        ConvertMeasure
    >(patterns.getContext(), &mapping);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQIRPass() {
    return std::make_unique<ConvertQuantumToQIRPass>();
}


