/// Implements the ConvertQIRToQuantumPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/QIRToQuantum/QIRToQuantum.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRBase.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRTypes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <algorithm>
#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOQUANTUM
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

Value getLastUsage(Value value)
{
    mlir::Value current = value;

    while (true) {
        auto uses = current.getUses();
        auto it = uses.begin();
        if (it == uses.end()) return current;

        auto &operand = *it;
        auto user = operand.getOwner();
        unsigned operandIndex = operand.getOperandNumber();
        current = user->getResult(operandIndex);
    }
}

struct ConvertQIRToQuantumPass
        : mlir::impl::ConvertQIRToQuantumBase<ConvertQIRToQuantumPass> {
    using ConvertQIRToQuantumBase::ConvertQIRToQuantumBase;

    void runOnOperation() override;
};

template<typename Op>
struct QIRToQuantumOpConversionPattern : OpConversionPattern<Op> {
    IRMapping* mapping;

    QIRToQuantumOpConversionPattern(
        TypeConverter &typeConverter,
        MLIRContext* ctx,
        IRMapping* mapping)
            : OpConversionPattern<Op>(typeConverter, ctx, 1),
              mapping(mapping)
    {}
};

struct ConvertFuncFunc : public QIRToQuantumOpConversionPattern<func::FuncOp> {
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

struct ConvertFuncReturn
        : public QIRToQuantumOpConversionPattern<func::ReturnOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op,
        func::ReturnOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Value> inputs;
        for (auto operand : adaptor.getOperands())
            inputs.emplace_back(mapping->lookup(operand));

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
        mapping->map(allocOp.getResult(), allocOp.getResult());
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
        Value newLhs = mapping->lookup(adaptor.getLhs());
        Value newRhs = mapping->lookup(adaptor.getRhs());
        auto swapOp =
            rewriter.create<quantum::SWAPOp>(op.getLoc(), newLhs, newRhs);
        mapping->map(adaptor.getLhs(), swapOp.getResult1());
        mapping->map(adaptor.getRhs(), swapOp.getResult2());
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
        auto input = this->mapping->lookup(adaptor.getInput());
        auto genOp =
            rewriter.create<TargetOp>(op.getLoc(), input, adaptor.getAngle());
        this->mapping->map(adaptor.getInput(), genOp.getResult());
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
        auto input = this->mapping->lookup(adaptor.getInput());
        auto genOp = rewriter.create<TargetOp>(op.getLoc(), input);
        this->mapping->map(adaptor.getInput(), genOp.getResult());
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
        auto ctrl = mapping->lookup(adaptor.getControl());
        auto tgt = mapping->lookup(adaptor.getTarget());
        auto cxOp = rewriter.create<quantum::CNOTOp>(op.getLoc(), ctrl, tgt);
        mapping->map(adaptor.getControl(), cxOp.getControlOut());
        mapping->map(adaptor.getTarget(), cxOp.getTargetOut());

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
        auto ctrl = mapping->lookup(adaptor.getControl());
        auto tgt = mapping->lookup(adaptor.getTarget());
        auto czOp = rewriter.create<quantum::CZOp>(op.getLoc(), ctrl, tgt);
        mapping->map(adaptor.getControl(), czOp.getControlOut());
        mapping->map(adaptor.getTarget(), czOp.getTargetOut());
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
        auto ctrl1 = mapping->lookup(adaptor.getControl1());
        auto ctrl2 = mapping->lookup(adaptor.getControl2());
        auto tgt = mapping->lookup(adaptor.getTarget());
        auto ccxOp =
            rewriter.create<quantum::CCXOp>(op.getLoc(), ctrl1, ctrl2, tgt);
        mapping->map(adaptor.getControl1(), ccxOp.getControl1Out());
        mapping->map(adaptor.getControl2(), ccxOp.getControl2Out());
        mapping->map(adaptor.getTarget(), ccxOp.getTargetOut());
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
        SmallVector<Value> inputs;
        for (auto operand : adaptor.getInput())
            inputs.emplace_back(mapping->lookup(operand));

        SmallVector<Type> resultTypes(
            inputs.size(),
            quantum::QubitType::get(getContext(), 1));

        auto barrierOp = rewriter.create<quantum::BarrierOp>(
            op.getLoc(),
            resultTypes,
            inputs);

        for (auto [input, result] :
             llvm::zip_equal(adaptor.getInput(), barrierOp.getResult()))
            mapping->map(input, result);

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
        auto input = mapping->lookup(adaptor.getInput());
        auto newOp = rewriter.create<quantum::U3Op>(
            op.getLoc(),
            input,
            adaptor.getTheta(),
            adaptor.getPhi(),
            adaptor.getLambda());
        mapping->map(adaptor.getInput(), newOp.getResult());
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
        auto input = mapping->lookup(adaptor.getInput());
        auto u1Op = rewriter.create<quantum::U1Op>(
            op.getLoc(),
            input,
            adaptor.getLambda());
        mapping->map(adaptor.getInput(), u1Op.getResult());
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
        auto input = mapping->lookup(adaptor.getInput());
        auto u2Op = rewriter.create<quantum::U2Op>(
            op.getLoc(),
            input,
            adaptor.getPhi(),
            adaptor.getLambda());
        mapping->map(adaptor.getInput(), u2Op.getResult());
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
        auto controlQubit = mapping->lookup(adaptor.getControl());
        auto targetQubit = mapping->lookup(adaptor.getTarget());
        auto angle = adaptor.getAngle();

        auto cryOp = rewriter.create<quantum::CRyOp>(
            op.getLoc(),
            controlQubit,
            targetQubit,
            angle);

        // Update the qubit map with outputs
        mapping->map(adaptor.getControl(), cryOp.getControlOut());
        mapping->map(adaptor.getTarget(), cryOp.getTargetOut());

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
        auto controlQubit = mapping->lookup(adaptor.getControl());
        auto targetQubit = mapping->lookup(adaptor.getTarget());
        auto angle = adaptor.getAngle();

        auto crzOp = rewriter.create<quantum::CRzOp>(
            op.getLoc(),
            controlQubit,
            targetQubit,
            angle);

        // Update the qubit map with outputs
        mapping->map(adaptor.getControl(), crzOp.getControlOut());
        mapping->map(adaptor.getTarget(), crzOp.getTargetOut());

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
        auto in = mapping->lookup(adaptor.getInput());
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
        auto input = mapping->lookup(adaptor.getInput());
        auto loc = op.getLoc();

        auto i1Type = rewriter.getI1Type();
        auto tensorType = mlir::RankedTensorType::get({1}, i1Type);

        auto genMeasureOp = rewriter.create<quantum::MeasureOp>(
            loc,
            tensorType,
            input.getType(),
            input);

        mapping->map(adaptor.getInput(), genMeasureOp.getResult());

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

struct ConvertGateOp : public QIRToQuantumOpConversionPattern<qir::GateOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        GateOp op,
        GateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> types;
        if (failed(
                getTypeConverter()->convertTypes(op.getArgumentTypes(), types)))
            op.emitOpError("Gate argument type conversion failed");

        FunctionType ftype = FunctionType::get(getContext(), types, types);
        auto gateOp = rewriter.create<quantum::GateOp>(
            op->getLoc(),
            op.getSymName(),
            op.getArgAttrs().value_or(ArrayAttr()),
            op.getResAttrs().value_or(ArrayAttr()),
            ftype);

        Block* newEntryBlock = gateOp.addEntryBlock();
        Block &oldEntryBlock = op.getBody().front();

        for (auto [oldArg, newArg] : llvm::zip(
                 oldEntryBlock.getArguments(),
                 newEntryBlock->getArguments())) {
            mapping->map(oldArg, newArg);
            mapping->map(newArg, newArg);
        }
        rewriter.setInsertionPointToStart(newEntryBlock);
        for (Operation &op : oldEntryBlock.without_terminator())
            rewriter.clone(op, *mapping);

        Operation* terminator = oldEntryBlock.getTerminator();
        rewriter.clone(*terminator, *mapping);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertGateOp

struct ConvertGateReturnOp
        : public QIRToQuantumOpConversionPattern<qir::ReturnOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        ReturnOp op,
        ReturnOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto gate = op->getParentOfType<quantum::GateOp>();
        if (!gate) op.emitOpError("Failed to access enclosing GateOp");

        auto &entryBlock = gate.getBody().front();
        SmallVector<Value> results;
        for (auto arg : entryBlock.getArguments())
            results.push_back(getLastUsage(arg));
        rewriter.create<quantum::ReturnOp>(op->getLoc(), results);
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertGateReturnOp

struct ConvertGateCallOp
        : public QIRToQuantumOpConversionPattern<qir::GateCallOp> {
    using QIRToQuantumOpConversionPattern::QIRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        GateCallOp op,
        GateCallOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Value> args;
        for (auto arg : adaptor.getOperands())
            args.push_back(mapping->lookup(arg));

        SmallVector<Type> resultTypes;
        if (failed(getTypeConverter()->convertTypes(
                op->getOperandTypes(),
                resultTypes)))
            op.emitOpError("Failed to convert GateCallOp result types");

        auto callOp = rewriter.create<quantum::GateCallOp>(
            op->getLoc(),
            adaptor.getCallee(),
            resultTypes,
            args);

        for (auto [arg, result] :
             llvm::zip_equal(adaptor.getOperands(), callOp->getResults()))
            mapping->map(arg, result);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertGateCallOp

} // namespace

void ConvertQIRToQuantumPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    IRMapping mapping;

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

    qir::populateConvertQIRToQuantumPatterns(typeConverter, patterns, mapping);

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
    IRMapping &mapping)
{
    patterns.add<
        ConvertFuncFunc,
        ConvertFuncReturn,
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
        ConvertReset,
        ConvertGateOp,
        ConvertGateReturnOp,
        ConvertGateCallOp>(typeConverter, patterns.getContext(), &mapping);
}

std::unique_ptr<Pass> mlir::createConvertQIRToQuantumPass()
{
    return std::make_unique<ConvertQIRToQuantumPass>();
}
