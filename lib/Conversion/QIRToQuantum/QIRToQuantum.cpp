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
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
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

class QubitMap {
public:
    QubitMap(MLIRContext* ctx) : ctx(ctx), qubits() {}

    void insert(Value use, Value gen) { qubits[use] = gen; }

    Value find(Value use) { return qubits[use]; }

    MLIRContext* getContext() const { return ctx; }

private:
    [[maybe_unused]] MLIRContext* ctx;
    llvm::DenseMap<Value, Value> qubits;
}; // class QubitMap

namespace {

struct ConvertQIRToQuantumPass
        : mlir::impl::ConvertQIRToQuantumBase<ConvertQIRToQuantumPass> {
    using ConvertQIRToQuantumBase::ConvertQIRToQuantumBase;

    void runOnOperation() override;
};

template<typename Op>
struct QIRToQuantumOpConversionPattern : OpConversionPattern<Op> {
    QubitMap* qubitMap;

    QIRToQuantumOpConversionPattern(
        TypeConverter &typeConverter,
        MLIRContext* ctx,
        QubitMap* qubitMap)
            : OpConversionPattern<Op>(typeConverter, ctx, 1),
              qubitMap(qubitMap)
    {}
};

struct ConvertAlloc : public QIRToQuantumOpConversionPattern<qir::AllocOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto allocOp = rewriter.replaceOpWithNewOp<quantum::AllocOp>(
            op,
            quantum::QubitType::get(getContext(), 1));
        qubitMap->insert(allocOp.getResult(), allocOp.getResult());
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertResultAlloc
        : public QIRToQuantumOpConversionPattern<qir::AllocResultOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

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

struct ConvertSwap : public QIRToQuantumOpConversionPattern<qir::SwapOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::SwapOp op,
        qir::SwapOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Retrieve the two input qubits from the adaptor.
        Value newLhs = qubitMap->find(adaptor.getLhs());
        Value newRhs = qubitMap->find(adaptor.getRhs());
        auto swapOp =
            rewriter.create<quantum::SWAPOp>(op.getLoc(), newLhs, newRhs);
        qubitMap->insert(adaptor.getLhs(), swapOp.getResult1());
        qubitMap->insert(adaptor.getRhs(), swapOp.getResult2());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertSwapOp

template<typename SourceOp, typename TargetOp>
struct ConvertRotation : public QIRToQuantumOpConversionPattern<SourceOp> {
    using QIRToQuantumOpConversionPattern<
        SourceOp>::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = this->qubitMap->find(adaptor.getInput());
        auto genOp =
            rewriter.create<TargetOp>(op.getLoc(), input, adaptor.getAngle());
        this->qubitMap->insert(adaptor.getInput(), genOp.getResult());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertRotationOp

template<typename SourceOp, typename TargetOp>
struct ConvertUnaryOp : public QIRToQuantumOpConversionPattern<SourceOp> {
    using QIRToQuantumOpConversionPattern<
        SourceOp>::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = this->qubitMap->find(adaptor.getInput());
        auto genOp = rewriter.create<TargetOp>(op.getLoc(), input);
        this->qubitMap->insert(adaptor.getInput(), genOp.getResult());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertUnaryOp

struct ConvertReset : public QIRToQuantumOpConversionPattern<qir::ResetOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        ResetOp op,
        ResetOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto in = qubitMap->find(adaptor.getInput());
        rewriter.replaceOpWithNewOp<quantum::DeallocateOp>(op, in);
        return success();
    }
}; // struct ConvertResetOp

struct ConvertMeasure : public QIRToQuantumOpConversionPattern<qir::MeasureOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = qubitMap->find(adaptor.getInput());
        auto loc = op.getLoc();

        auto i1Type = rewriter.getI1Type();
        auto tensorType = mlir::RankedTensorType::get({1}, i1Type);

        auto genMeasureOp = rewriter.create<quantum::MeasureOp>(
            loc,
            tensorType,
            input.getType(),
            input);

        qubitMap->insert(adaptor.getInput(), genMeasureOp.getResult());

        // qir.measure (%q, %r)
        // Find uses of %r and get %m of
        // %m = qir.read_measurement (%r)
        auto measuredRegister = op.getResult();
        auto uses = measuredRegister.getUses();
        for (auto it = uses.begin(); it != uses.end(); ++it) {
            auto otherOp = it.getOperand()->getOwner();
            if (auto readOp = llvm::dyn_cast<qir::ReadMeasurementOp>(otherOp)) {
                // Replace usages of %m with new measurement result
                readOp.getMeasurement().replaceAllUsesWith(
                    genMeasureOp.getMeasurement());
                rewriter.eraseOp(readOp);
            }
        }
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertMeasureOp

struct ConvertReadMeasurement
        : public QIRToQuantumOpConversionPattern<qir::ReadMeasurementOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        ReadMeasurementOp op,
        ReadMeasurementOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        return op->emitOpError(
            "ReadMeasurement should already have been removed");
    }
}; // struct ConvertReadMeasurementOp

} // namespace

void ConvertQIRToQuantumPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QubitMap qubitMap(context);

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

    qir::populateConvertQIRToQuantumPatterns(typeConverter, patterns, qubitMap);

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
    RewritePatternSet &patterns,
    QubitMap &qubitMap)
{
    patterns.add<
        ConvertAlloc,
        ConvertSwap,
        ConvertResultAlloc,
        ConvertUnaryOp<qir::HOp, quantum::HOp>,
        ConvertUnaryOp<qir::XOp, quantum::XOp>,
        ConvertRotation<qir::RzOp, quantum::RzOp>,
        ConvertMeasure,
        ConvertReset>(typeConverter, patterns.getContext(), &qubitMap);
}

std::unique_ptr<Pass> mlir::createConvertQIRToQuantumPass()
{
    return std::make_unique<ConvertQIRToQuantumPass>();
}
