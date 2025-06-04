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

#include <algorithm>
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

    SmallVector<Value> findAll(ValueRange uses)
    {
        llvm::SmallVector<Value> results;
        std::for_each(uses.begin(), uses.end(), [&](Value v) {
            results.push_back(find(v));
        });
        return results;
    }

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

struct ConvertFunc : public QIRToQuantumOpConversionPattern<func::FuncOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

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

struct ConvertReturn : public QIRToQuantumOpConversionPattern<func::ReturnOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op,
        func::ReturnOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto inputs = qubitMap->findAll(adaptor.getOperands());
        rewriter.create<func::ReturnOp>(op->getLoc(), inputs);
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertReturn

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

struct ConvertCNOT : public QIRToQuantumOpConversionPattern<qir::CNOTOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::CNOTOp op,
        qir::CNOTOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctrl = qubitMap->find(adaptor.getControl());
        auto tgt = qubitMap->find(adaptor.getTarget());
        auto cxOp = rewriter.create<quantum::CNOTOp>(
            op.getLoc(),
            ValueRange{ctrl, tgt});
        qubitMap->insert(adaptor.getControl(), cxOp.getControlOut());
        qubitMap->insert(adaptor.getTarget(), cxOp.getTargetOut());

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertCZ : public QIRToQuantumOpConversionPattern<qir::CZOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;
    LogicalResult matchAndRewrite(
        qir::CZOp op,
        qir::CZOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctrl = qubitMap->find(adaptor.getControl());
        auto tgt = qubitMap->find(adaptor.getTarget());
        auto czOp = rewriter.create<quantum::CZOp>(op.getLoc(), ctrl, tgt);
        qubitMap->insert(adaptor.getControl(), czOp.getControlOut());
        qubitMap->insert(adaptor.getTarget(), czOp.getTargetOut());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertCZOp

struct ConvertCCX : public QIRToQuantumOpConversionPattern<qir::CCXOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;
    LogicalResult matchAndRewrite(
        qir::CCXOp op,
        qir::CCXOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctrl1 = qubitMap->find(adaptor.getControl1());
        auto ctrl2 = qubitMap->find(adaptor.getControl2());
        auto tgt = qubitMap->find(adaptor.getTarget());
        auto ccxOp =
            rewriter.create<quantum::CCXOp>(op.getLoc(), ctrl1, ctrl2, tgt);
        qubitMap->insert(adaptor.getControl1(), ccxOp.getControl1Out());
        qubitMap->insert(adaptor.getControl2(), ccxOp.getControl2Out());
        qubitMap->insert(adaptor.getTarget(), ccxOp.getTargetOut());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertCCXOp

struct ConvertBarrierOp
        : public QIRToQuantumOpConversionPattern<qir::BarrierOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::BarrierOp op,
        qir::BarrierOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        ValueRange qirQubits = adaptor.getInput();
        Value quantumQubit = qubitMap->find(qirQubits.front());
        auto barrierOp = rewriter.create<quantum::BarrierOp>(
            op.getLoc(),
            quantumQubit.getType(),
            quantumQubit);

        qubitMap->insert(qirQubits.front(), barrierOp.getResult());

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertBarrierOp

struct ConvertU3 : public QIRToQuantumOpConversionPattern<qir::U3Op> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::U3Op op,
        qir::U3OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = qubitMap->find(adaptor.getInput());
        auto newOp = rewriter.create<quantum::U3Op>(
            op.getLoc(),
            input,
            adaptor.getTheta(),
            adaptor.getPhi(),
            adaptor.getLambda());
        qubitMap->insert(adaptor.getInput(), newOp.getResult());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertU3Op

struct ConvertU1 : public QIRToQuantumOpConversionPattern<qir::U1Op> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::U1Op op,
        qir::U1OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = qubitMap->find(adaptor.getInput());
        auto u1Op = rewriter.create<quantum::U1Op>(
            op.getLoc(),
            input,
            adaptor.getLambda());
        qubitMap->insert(adaptor.getInput(), u1Op.getResult());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertU1Op

struct ConvertU2 : public QIRToQuantumOpConversionPattern<qir::U2Op> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::U2Op op,
        qir::U2OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = qubitMap->find(adaptor.getInput());
        auto u2Op = rewriter.create<quantum::U2Op>(
            op.getLoc(),
            input,
            adaptor.getPhi(),
            adaptor.getLambda());
        qubitMap->insert(adaptor.getInput(), u2Op.getResult());
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertU2Op

struct ConvertCRy : public QIRToQuantumOpConversionPattern<qir::CRyOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::CRyOp op,
        qir::CRyOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlQubit = qubitMap->find(adaptor.getControl());
        auto targetQubit = qubitMap->find(adaptor.getTarget());
        auto angle = adaptor.getAngle();

        auto cryOp = rewriter.create<quantum::CRyOp>(
            op.getLoc(),
            controlQubit,
            targetQubit,
            angle);

        // Update the qubit map with outputs
        qubitMap->insert(adaptor.getControl(), cryOp.getControlOut());
        qubitMap->insert(adaptor.getTarget(), cryOp.getTargetOut());

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertCRz : public QIRToQuantumOpConversionPattern<qir::CRzOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qir::CRzOp op,
        qir::CRzOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlQubit = qubitMap->find(adaptor.getControl());
        auto targetQubit = qubitMap->find(adaptor.getTarget());
        auto angle = adaptor.getAngle();

        auto crzOp = rewriter.create<quantum::CRzOp>(
            op.getLoc(),
            controlQubit,
            targetQubit,
            angle);

        // Update the qubit map with outputs
        qubitMap->insert(adaptor.getControl(), crzOp.getControlOut());
        qubitMap->insert(adaptor.getTarget(), crzOp.getTargetOut());

        rewriter.eraseOp(op);
        return success();
    }
};

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
    target.addLegalDialect<quantum::QuantumDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
        auto types = op.getOperandTypes();
        bool legal = true;
        std::for_each(types.begin(), types.end(), [&](auto ty) {
            legal &= typeConverter.isLegal(ty);
        });
        return legal;
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
        ConvertFunc,
        ConvertReturn,
        ConvertAlloc,
        ConvertSwap,
        ConvertResultAlloc,
        ConvertUnaryOp<qir::HOp, quantum::HOp>,
        ConvertUnaryOp<qir::XOp, quantum::XOp>,
        ConvertUnaryOp<qir::YOp, quantum::YOp>,
        ConvertUnaryOp<qir::ZOp, quantum::ZOp>,
        ConvertRotation<qir::RzOp, quantum::RzOp>,
        ConvertRotation<qir::RxOp, quantum::RxOp>,
        ConvertRotation<qir::RyOp, quantum::RyOp>,
        ConvertUnaryOp<qir::SOp, quantum::SOp>,
        ConvertUnaryOp<qir::TOp, quantum::TOp>,
        ConvertUnaryOp<qir::SdgOp, quantum::SdgOp>,
        ConvertUnaryOp<qir::TdgOp, quantum::TdgOp>,
        ConvertCNOT,
        ConvertCZ,
        ConvertCCX,
        ConvertU3,
        ConvertU2,
        ConvertU1,
        ConvertCRy,
        ConvertCRz,
        ConvertBarrierOp,
        ConvertMeasure,
        ConvertReset>(typeConverter, patterns.getContext(), &qubitMap);
}

std::unique_ptr<Pass> mlir::createConvertQIRToQuantumPass()
{
    return std::make_unique<ConvertQIRToQuantumPass>();
}
