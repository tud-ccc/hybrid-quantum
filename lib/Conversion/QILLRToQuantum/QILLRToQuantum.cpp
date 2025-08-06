/// Implements the ConvertQILLRToQuantumPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/QILLRToQuantum/QILLRToQuantum.h"

#include "mlir/Dialect/SCF/IR/SCFOpsDialect.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.h"
#include "quantum-mlir/Dialect/QQT/IR/QQT.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTBase.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTTypes.h"
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
using namespace mlir::qqt;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQILLRTOQUANTUM
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

struct ConvertQILLRToQuantumPass
        : mlir::impl::ConvertQILLRToQuantumBase<ConvertQILLRToQuantumPass> {
    using ConvertQILLRToQuantumBase::ConvertQILLRToQuantumBase;

    void runOnOperation() override;
};

template<typename Op>
struct QILLRToQuantumOpConversionPattern : OpConversionPattern<Op> {
    /// Maps the `qillr.Qubit` to the `qqt.QubitRef`
    IRMapping* mapping;

    QILLRToQuantumOpConversionPattern(
        TypeConverter &typeConverter,
        MLIRContext* ctx,
        IRMapping* mapping)
            : OpConversionPattern<Op>(typeConverter, ctx, 1),
              mapping(mapping)
    {}
};

struct ConvertAlloc : public QILLRToQuantumOpConversionPattern<qillr::AllocOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::AllocOp op,
        qillr::AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Create qubit reference value and store mapping from old qubit
        // refrence to the fresh created value
        auto promoteOp = rewriter.create<qqt::PromoteOp>(op->getLoc());
        mapping->map(op.getResult(), promoteOp.getResult());

        auto allocOp = rewriter.create<quantum::AllocOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1));

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            allocOp.getResult(),
            promoteOp.getResult());

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertResultAlloc
        : public QILLRToQuantumOpConversionPattern<qillr::AllocResultOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::AllocResultOp op,
        qillr::AllocResultOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // We do not have a representation for result registers in Quantum
        // dialect
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertResultAllocOp

struct ConvertSwap : public QILLRToQuantumOpConversionPattern<qillr::SwapOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::SwapOp op,
        qillr::SwapOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Value lhsRef = mapping->lookup(adaptor.getLhs());
        Value rhsRef = mapping->lookup(adaptor.getRhs());
        auto loadLhsOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1),
            lhsRef);
        auto loadRhsOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1),
            rhsRef);

        auto swapOp = rewriter.create<quantum::SWAPOp>(
            op.getLoc(),
            loadLhsOp.getResult(),
            loadRhsOp.getResult());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            swapOp.getResultLhs(),
            lhsRef);
        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            swapOp.getResultRhs(),
            rhsRef);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertSwapOp

template<typename SourceOp, typename TargetOp>
struct ConvertRotation : public QILLRToQuantumOpConversionPattern<SourceOp> {
    using QILLRToQuantumOpConversionPattern<
        SourceOp>::QILLRToQuantumOpConversionPattern;

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
struct ConvertUnaryOp : public QILLRToQuantumOpConversionPattern<SourceOp> {
    using QILLRToQuantumOpConversionPattern<
        SourceOp>::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = this->mapping->lookup(adaptor.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op.getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            input);

        auto genOp = rewriter.create<TargetOp>(op.getLoc(), loadOp.getResult());

        rewriter.create<qqt::StoreOp>(op.getLoc(), genOp.getResult(), input);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertUnaryOp

struct ConvertCNOT : public QILLRToQuantumOpConversionPattern<qillr::CNOTOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::CNOTOp op,
        qillr::CNOTOpAdaptor adaptor,
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

struct ConvertCZ : public QILLRToQuantumOpConversionPattern<qillr::CZOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;
    LogicalResult matchAndRewrite(
        qillr::CZOp op,
        qillr::CZOpAdaptor adaptor,
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

struct ConvertCCX : public QILLRToQuantumOpConversionPattern<qillr::CCXOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;
    LogicalResult matchAndRewrite(
        qillr::CCXOp op,
        qillr::CCXOpAdaptor adaptor,
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
        : public QILLRToQuantumOpConversionPattern<qillr::BarrierOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::BarrierOp op,
        qillr::BarrierOpAdaptor adaptor,
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

struct ConvertU3 : public QILLRToQuantumOpConversionPattern<qillr::U3Op> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::U3Op op,
        qillr::U3OpAdaptor adaptor,
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

struct ConvertU1 : public QILLRToQuantumOpConversionPattern<qillr::U1Op> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::U1Op op,
        qillr::U1OpAdaptor adaptor,
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

struct ConvertU2 : public QILLRToQuantumOpConversionPattern<qillr::U2Op> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::U2Op op,
        qillr::U2OpAdaptor adaptor,
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

struct ConvertCRy : public QILLRToQuantumOpConversionPattern<qillr::CRyOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::CRyOp op,
        qillr::CRyOpAdaptor adaptor,
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

struct ConvertCRz : public QILLRToQuantumOpConversionPattern<qillr::CRzOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::CRzOp op,
        qillr::CRzOpAdaptor adaptor,
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

struct ConvertCU1 : public QILLRToQuantumOpConversionPattern<qillr::CU1Op> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::CU1Op op,
        qillr::CU1OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlQubit = mapping->lookup(adaptor.getControl());
        auto targetQubit = mapping->lookup(adaptor.getTarget());
        auto angle = adaptor.getAngle();

        auto cu1op = rewriter.create<quantum::CU1Op>(
            op.getLoc(),
            controlQubit,
            targetQubit,
            angle);

        // Update the qubit map with outputs
        mapping->map(adaptor.getControl(), cu1op.getControlOut());
        mapping->map(adaptor.getTarget(), cu1op.getTargetOut());

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertCSwap : public QILLRToQuantumOpConversionPattern<qillr::CSwapOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::CSwapOp op,
        qillr::CSwapOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlQubit = mapping->lookup(adaptor.getControl());
        auto lhs = mapping->lookup(adaptor.getLhs());
        auto rhs = mapping->lookup(adaptor.getRhs());

        auto cswap = rewriter.create<quantum::CSWAPOp>(
            op.getLoc(),
            controlQubit,
            lhs,
            rhs);

        // Update the qubit map with outputs
        mapping->map(adaptor.getControl(), cswap.getControlOut());
        mapping->map(adaptor.getLhs(), cswap.getLhsOut());
        mapping->map(adaptor.getRhs(), cswap.getRhsOut());

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertReset : public QILLRToQuantumOpConversionPattern<qillr::ResetOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::ResetOp op,
        qillr::ResetOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto in = mapping->lookup(adaptor.getInput());
        rewriter.replaceOpWithNewOp<quantum::DeallocateOp>(op, in);
        return success();
    }
}; // struct ConvertResetOp

struct ConvertMeasure
        : public QILLRToQuantumOpConversionPattern<qillr::MeasureOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::MeasureOp op,
        qillr::MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto input = mapping->lookup(adaptor.getInput());
        auto loc = op.getLoc();

        auto genMeasureOp =
            rewriter.create<quantum::MeasureSingleOp>(loc, input);

        mapping->map(adaptor.getInput(), genMeasureOp.getResult());

        // qillr.measure (%q, %r)
        // Find uses of %r and get %m of
        // %m = qillr.read_measurement (%r)
        auto measuredRegister = op.getResult();
        auto uses = measuredRegister.getUses();
        for (auto it = uses.begin(); it != uses.end(); ++it) {
            auto otherOp = it.getOperand()->getOwner();
            if (auto readOp =
                    llvm::dyn_cast<qillr::ReadMeasurementOp>(otherOp)) {
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
        : public QILLRToQuantumOpConversionPattern<qillr::ReadMeasurementOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        qillr::ReadMeasurementOp op,
        qillr::ReadMeasurementOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        return op->emitOpError(
            "ReadMeasurement should already have been removed");
    }
}; // struct ConvertReadMeasurementOp

} // namespace

void ConvertQILLRToQuantumPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    IRMapping mapping;

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](qillr::QubitType ty) {
        // return quantum::QubitType::get(ty.getContext(), 1);
        return qqt::QubitRefType::get(ty.getContext());
    });

    qqt::populateConvertQILLRToQuantumPatterns(
        typeConverter,
        patterns,
        mapping);

    target.addIllegalDialect<qillr::QILLRDialect>();
    target.addLegalDialect<quantum::QuantumDialect>();
    target.addLegalDialect<qqt::QQTDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        return signalPassFailure();
}

void mlir::qqt::populateConvertQILLRToQuantumPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    IRMapping &mapping)
{
    patterns.add<
        ConvertAlloc,
        ConvertSwap,
        ConvertResultAlloc,
        ConvertUnaryOp<qillr::HOp, quantum::HOp>,
        ConvertUnaryOp<qillr::SXOp, quantum::SXOp>,
        ConvertUnaryOp<qillr::XOp, quantum::XOp>,
        ConvertUnaryOp<qillr::YOp, quantum::YOp>,
        ConvertUnaryOp<qillr::ZOp, quantum::ZOp>,
        ConvertUnaryOp<qillr::IdOp, quantum::IdOp>,
        ConvertRotation<qillr::RzOp, quantum::RzOp>,
        ConvertRotation<qillr::RxOp, quantum::RxOp>,
        ConvertRotation<qillr::RyOp, quantum::RyOp>,
        ConvertRotation<qillr::PhaseOp, quantum::PhaseOp>,
        ConvertUnaryOp<qillr::SOp, quantum::SOp>,
        ConvertUnaryOp<qillr::TOp, quantum::TOp>,
        ConvertUnaryOp<qillr::SdgOp, quantum::SdgOp>,
        ConvertUnaryOp<qillr::TdgOp, quantum::TdgOp>,
        ConvertCNOT,
        ConvertCZ,
        ConvertCCX,
        ConvertU3,
        ConvertU2,
        ConvertU1,
        ConvertCRy,
        ConvertCRz,
        ConvertCU1,
        ConvertBarrierOp,
        ConvertMeasure,
        ConvertReset>(typeConverter, patterns.getContext(), &mapping);
}

std::unique_ptr<Pass> mlir::createConvertQILLRToQuantumPass()
{
    return std::make_unique<ConvertQILLRToQuantumPass>();
}
