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
#include "quantum-mlir/Dialect/QQT/IR/QQTBase.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTTypes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace mlir::qqt;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQILLRTOQUANTUM
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

qillr::MeasureOp findImmediateDominatingMeasure(
    qillr::ReadMeasurementOp readMeasurement,
    DominanceInfo &domInfo)
{
    Operation* op = readMeasurement.getOperation();
    Block* block = op->getBlock();
    Region* region = block->getParent();

    // Case 1: single-block region -> just scan backwards
    if (region->hasOneBlock()) {
        for (Operation* it = op->getPrevNode(); it; it = it->getPrevNode())
            if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(it))
                if (measure.getResult() == readMeasurement.getInput())
                    return measure;
        return nullptr;
    }

    // Case 2: multiple blocks -> use dominator tree
    for (Operation* it = op->getPrevNode(); it; it = it->getPrevNode())
        if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(it))
            if (measure.getResult() == readMeasurement.getInput())
                return measure;

    auto* node = domInfo.getNode(block);
    while (auto* idom = node->getIDom()) {
        Block* idomBlock = idom->getBlock();
        for (auto &blockOp : llvm::reverse(*idomBlock))
            if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(&blockOp))
                if (measure.getResult() == readMeasurement.getInput())
                    return measure;
        node = idom;
    }

    return nullptr;
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
    DominanceInfo &domInfo;

    QILLRToQuantumOpConversionPattern(
        TypeConverter &typeConverter,
        MLIRContext* ctx,
        IRMapping* mapping,
        DominanceInfo &domInfo)
            : OpConversionPattern<Op>(typeConverter, ctx, 1),
              mapping(mapping),
              domInfo(domInfo)
    {}

    llvm::MapVector<Value, Value> rallocToRead;
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
        // There is no representation for result registers in Quantum
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
        Value lhsRef = mapping->lookup(op.getLhs());
        Value rhsRef = mapping->lookup(op.getRhs());
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
        auto qubitRef = this->mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            qubitRef);

        auto genOp = rewriter.create<TargetOp>(
            op.getLoc(),
            loadOp.getResult(),
            adaptor.getAngle());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            genOp.getResult(),
            qubitRef);

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
        auto qubitRef = this->mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op.getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            qubitRef);

        auto genOp = rewriter.create<TargetOp>(op.getLoc(), loadOp.getResult());

        rewriter.create<qqt::StoreOp>(op.getLoc(), genOp.getResult(), qubitRef);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertUnaryOp

template<typename ControledSourceOp, typename ControledTargetOp>
struct ConvertControledOp
        : public QILLRToQuantumOpConversionPattern<ControledSourceOp> {
    using QILLRToQuantumOpConversionPattern<
        ControledSourceOp>::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        ControledSourceOp op,
        OpConversionPattern<ControledSourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlRef = this->mapping->lookup(op.getControl());
        auto targetRef = this->mapping->lookup(op.getTarget());

        auto loadControlOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            controlRef);
        auto loadTargetOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            targetRef);

        auto cxOp = rewriter.create<ControledTargetOp>(
            op.getLoc(),
            loadControlOp.getResult(),
            loadTargetOp.getResult());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            cxOp.getControlOut(),
            controlRef);
        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            cxOp.getTargetOut(),
            targetRef);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertCCX : public QILLRToQuantumOpConversionPattern<qillr::CCXOp> {
    using QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;
    LogicalResult matchAndRewrite(
        qillr::CCXOp op,
        qillr::CCXOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctrl1Ref = mapping->lookup(op.getControl1());
        auto ctrl2Ref = mapping->lookup(op.getControl2());
        auto targetRef = mapping->lookup(op.getTarget());

        auto loadControl1Op = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            ctrl1Ref);
        auto loadControl2Op = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            ctrl2Ref);
        auto loadTargetOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            targetRef);

        auto ccxOp = rewriter.create<quantum::CCXOp>(
            op.getLoc(),
            loadControl1Op.getResult(),
            loadControl2Op.getResult(),
            loadTargetOp.getResult());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            ccxOp.getControl1Out(),
            ctrl1Ref);
        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            ccxOp.getControl2Out(),
            ctrl2Ref);
        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            ccxOp.getTargetOut(),
            targetRef);

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
        SmallVector<Value> refs;
        SmallVector<Value> inputs;
        for (auto in : op.getInput()) {
            auto qubitRef = refs.emplace_back(mapping->lookup(in));
            auto loadInputOp = rewriter.create<qqt::LoadOp>(
                op->getLoc(),
                quantum::QubitType::get(this->getContext(), 1),
                qubitRef);
            inputs.emplace_back(loadInputOp.getResult());
        }

        SmallVector<Type> resultTypes(
            inputs.size(),
            quantum::QubitType::get(getContext(), 1));

        auto barrierOp = rewriter.create<quantum::BarrierOp>(
            op.getLoc(),
            resultTypes,
            inputs);

        for (auto it : llvm::enumerate(barrierOp.getResult()))
            rewriter.create<qqt::StoreOp>(
                op->getLoc(),
                it.value(),
                refs[it.index()]);

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
        auto qubitRef = mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1),
            qubitRef);

        auto newOp = rewriter.create<quantum::U3Op>(
            op.getLoc(),
            loadOp.getResult(),
            adaptor.getTheta(),
            adaptor.getPhi(),
            adaptor.getLambda());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            newOp.getResult(),
            qubitRef);

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
        auto qubitRef = mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1),
            qubitRef);

        auto u1Op = rewriter.create<quantum::U1Op>(
            op.getLoc(),
            loadOp.getResult(),
            adaptor.getLambda());

        rewriter.create<qqt::StoreOp>(op->getLoc(), u1Op.getResult(), qubitRef);

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
        auto qubitRef = mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(getContext(), 1),
            qubitRef);

        auto u2Op = rewriter.create<quantum::U2Op>(
            op.getLoc(),
            loadOp.getResult(),
            adaptor.getPhi(),
            adaptor.getLambda());

        rewriter.create<qqt::StoreOp>(op->getLoc(), u2Op.getResult(), qubitRef);

        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertU2Op

template<typename SourceOp, typename TargetOp>
struct ConvertControledRotation
        : public QILLRToQuantumOpConversionPattern<SourceOp> {
    using QILLRToQuantumOpConversionPattern<
        SourceOp>::QILLRToQuantumOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto controlRef = this->mapping->lookup(op.getControl());
        auto targetRef = this->mapping->lookup(op.getTarget());

        auto loadControlOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            controlRef);
        auto loadTargetOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            targetRef);

        auto cryOp = rewriter.create<TargetOp>(
            op.getLoc(),
            loadControlOp.getResult(),
            loadTargetOp.getResult(),
            adaptor.getAngle());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            cryOp.getControlOut(),
            controlRef);
        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            cryOp.getTargetOut(),
            targetRef);

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
        auto controlRef = mapping->lookup(op.getControl());
        auto lhsRef = mapping->lookup(op.getLhs());
        auto rhsRef = mapping->lookup(op.getRhs());

        auto loadControlOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            controlRef);
        auto loadLhsOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            lhsRef);
        auto loadRhsOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            rhsRef);

        auto cswap = rewriter.create<quantum::CSWAPOp>(
            op.getLoc(),
            loadControlOp.getResult(),
            loadLhsOp.getResult(),
            loadRhsOp.getResult());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            cswap.getControlOut(),
            controlRef);
        rewriter.create<qqt::StoreOp>(op->getLoc(), cswap.getLhsOut(), lhsRef);
        rewriter.create<qqt::StoreOp>(op->getLoc(), cswap.getRhsOut(), rhsRef);

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
        auto inRef = mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            inRef);

        bool hasFollowerOps = false;
        for (auto otherOp : op.getInput().getUsers())
            if (otherOp != op && domInfo.properlyDominates(op, otherOp)) {
                hasFollowerOps = true;
                break;
            }

        if (hasFollowerOps) {
            auto resetOp = rewriter.create<quantum::ResetOp>(
                op->getLoc(),
                quantum::QubitType::get(this->getContext(), 1),
                loadOp.getResult());
            rewriter.create<qqt::StoreOp>(
                op->getLoc(),
                resetOp.getResult(),
                inRef);
        } else {
            rewriter.create<quantum::DeallocateOp>(
                op->getLoc(),
                loadOp.getResult());
            rewriter.create<qqt::DestructOp>(op->getLoc(), inRef);
        }

        rewriter.eraseOp(op);
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
        auto resultRegister = op.getResult();
        auto inRef = mapping->lookup(op.getInput());
        auto loadOp = rewriter.create<qqt::LoadOp>(
            op->getLoc(),
            quantum::QubitType::get(this->getContext(), 1),
            inRef);

        auto genMeasureOp = rewriter.create<quantum::MeasureSingleOp>(
            op->getLoc(),
            loadOp.getResult());

        rewriter.create<qqt::StoreOp>(
            op->getLoc(),
            genMeasureOp.getResult(),
            inRef);

        // qillr.measure (%q, %r)
        // Find uses of %r and get %m of
        // %m = qillr.read_measurement (%r)
        for (auto rallocUser : resultRegister.getUsers()) {
            if (auto read =
                    llvm::dyn_cast<qillr::ReadMeasurementOp>(rallocUser)) {
                if (op == findImmediateDominatingMeasure(read, domInfo)) {
                    read.getMeasurement().replaceAllUsesWith(
                        genMeasureOp.getMeasurement());
                    rewriter.eraseOp(read);
                }
            }
        }
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertMeasureOp

// struct ConvertReadMeasurement
//         : public QILLRToQuantumOpConversionPattern<qillr::ReadMeasurementOp>
//         {
//     using
//     QILLRToQuantumOpConversionPattern::QILLRToQuantumOpConversionPattern;

//     LogicalResult matchAndRewrite(
//         qillr::ReadMeasurementOp op,
//         qillr::ReadMeasurementOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         return llvm::success();
//     }
// }; // struct ConvertReadMeasurementOp

} // namespace

void ConvertQILLRToQuantumPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    IRMapping mapping;
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](qillr::QubitType ty) {
        // return quantum::QubitType::get(ty.getContext(), 1);
        return qqt::QubitRefType::get(ty.getContext());
    });

    qqt::populateConvertQILLRToQuantumPatterns(
        typeConverter,
        patterns,
        mapping,
        domInfo);

    target.addIllegalDialect<qillr::QILLRDialect>();
    target.addLegalDialect<quantum::QuantumDialect>();
    target.addLegalDialect<qqt::QQTDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        return signalPassFailure();

    return markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

void mlir::qqt::populateConvertQILLRToQuantumPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    IRMapping &mapping,
    DominanceInfo &domInfo)
{
    patterns.add<
        ConvertResultAlloc,
        // ConvertReadMeasurement,
        ConvertAlloc,
        ConvertSwap,
        ConvertCSwap,
        ConvertMeasure,
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
        ConvertControledOp<qillr::CNOTOp, quantum::CNOTOp>,
        ConvertControledOp<qillr::CZOp, quantum::CZOp>,
        ConvertCCX,
        ConvertU3,
        ConvertU2,
        ConvertU1,
        ConvertControledRotation<qillr::CRyOp, quantum::CRyOp>,
        ConvertControledRotation<qillr::CRzOp, quantum::CRzOp>,
        ConvertControledRotation<qillr::CU1Op, quantum::CU1Op>,
        ConvertBarrierOp,
        ConvertReset>(typeConverter, patterns.getContext(), &mapping, domInfo);
}

std::unique_ptr<Pass> mlir::createConvertQILLRToQuantumPass()
{
    return std::make_unique<ConvertQILLRToQuantumPass>();
}
