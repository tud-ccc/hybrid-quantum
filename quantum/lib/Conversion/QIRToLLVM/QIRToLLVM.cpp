/// Implements the ConvertQIRToLLVMPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)

#include "quantum-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"

#include <cstdint>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOLLVM
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir
//===----------------------------------------------------------------------===//

struct mlir::qir::AllocationAnalysis {

    AllocationAnalysis(Operation* op)
    {
        int64_t allocOpId = 0;
        int64_t allocResultOpId = 0;

        // Walk through all operations in the module and find AllocOp
        op->walk([&](AllocOp allocOp) { allocMapping[allocOp] = allocOpId++; });

        // Walk through all operations in the module and find AllocResultOp
        op->walk([&](AllocResultOp allocResultOp) {
            resultMapping[allocResultOp] = allocResultOpId++;
        });
    }

    // ensure that the counts are non-zero if there are any allocations
    bool verify() const
    {
        return (getQubitCount() >= 0) && (getResultCount() >= 0);
    }

    int64_t getQubitCount() const { return allocMapping.size(); }
    int64_t getResultCount() const { return resultMapping.size(); }

    int64_t getQubitId(AllocOp allocOp) const
    {
        auto it = allocMapping.find(allocOp);
        assert(it != allocMapping.end() && "AllocOp not found in mapping!");
        return it->second;
    }

    int64_t getResultId(AllocResultOp allocResultOp) const
    {
        auto it = resultMapping.find(allocResultOp);
        assert(
            it != resultMapping.end() && "AllocResultOp not found in mapping!");
        return it->second;
    }

private:
    llvm::DenseMap<AllocOp, int64_t> allocMapping;
    llvm::DenseMap<AllocResultOp, int64_t> resultMapping;
};

namespace {

struct ConvertQIRToLLVMPass
        : mlir::impl::ConvertQIRToLLVMBase<ConvertQIRToLLVMPass> {
    using ConvertQIRToLLVMBase::ConvertQIRToLLVMBase;

    void runOnOperation() override;
};

LLVM::LLVMFuncOp ensureFunctionDeclaration(
    PatternRewriter &rewriter,
    Operation* op,
    StringRef fnSymbol,
    Type fnType)
{
    Operation* fnDecl = SymbolTable::lookupNearestSymbolFrom(
        op,
        rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl =
            rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    } else {
        assert(
            isa<LLVM::LLVMFuncOp>(fnDecl)
            && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
};

struct AllocOpPattern : public ConvertOpToLLVMPattern<AllocOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    AllocOpPattern(
        LLVMTypeConverter &typeConverter,
        AllocationAnalysis &analysis)
            : ConvertOpToLLVMPattern(typeConverter),
              analysis(analysis)
    {}

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Get the unique ID for this AllocOp from the analysis result
        int64_t qubitId = analysis.getQubitId(op);

        // Create an LLVM constant integer to represent the unique ID.
        Type i64Type = rewriter.getI64Type();
        Value intValue = rewriter.create<LLVM::ConstantOp>(
            loc,
            i64Type,
            rewriter.getI64IntegerAttr(qubitId));

        // Create a pointer type.
        Type ptrType = LLVM::LLVMPointerType::get(ctx);

        // Create the inttoptr operation.
        Value ptrValue =
            rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, intValue);

        // Replace the original op with the computed pointer.
        rewriter.replaceOp(op, ptrValue);
        return success();
    }

private:
    AllocationAnalysis &analysis;
};

struct ReadMeasurementOpPattern
        : public ConvertOpToLLVMPattern<qir::ReadMeasurementOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        ReadMeasurementOp op,
        ReadMeasurementOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const StringRef qirName = "__quantum__qis__read_result__body";
        Type ptrType = LLVM::LLVMPointerType::get(getContext());
        Type i1Type = rewriter.getI1Type();
        Type qirSignature =
            LLVM::LLVMFunctionType::get(i1Type, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        // Get the qubit argument from the operation
        Value inputResult = adaptor.getInput();

        // Create the call operation to apply the Hadamard gate
        auto measureOp = rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            i1Type,
            fnDecl.getSymName(),
            ValueRange{inputResult});

        auto tensor = rewriter.create<mlir::tensor::FromElementsOp>(
            op.getLoc(),
            ValueRange{measureOp.getResult()});

        rewriter.replaceOp(op, tensor);

        return success();
    }
};

struct AllocResultOpPattern : public ConvertOpToLLVMPattern<AllocResultOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    AllocResultOpPattern(
        LLVMTypeConverter &typeConverter,
        AllocationAnalysis &analysis)
            : ConvertOpToLLVMPattern(typeConverter),
              analysis(analysis)
    {}

    LogicalResult matchAndRewrite(
        AllocResultOp op,
        AllocResultOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Get the unique ID for this AllocResultOp from the analysis result
        int64_t resultId = analysis.getResultId(op);

        // Create an LLVM constant integer to represent the unique ID.
        Type i64Type = rewriter.getI64Type();
        Value intValue = rewriter.create<LLVM::ConstantOp>(
            loc,
            i64Type,
            rewriter.getI64IntegerAttr(resultId));

        // Create a pointer type.
        Type ptrType = LLVM::LLVMPointerType::get(ctx);

        // Create the inttoptr operation.
        Value ptrValue =
            rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, intValue);

        // Replace the original op with the computed pointer.
        rewriter.replaceOp(op, ptrValue);
        return success();
    }

private:
    AllocationAnalysis &analysis;
};

struct HOpPattern : public ConvertOpToLLVMPattern<HOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        HOp op,
        HOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define the QIR function name for the Hadamard gate
        StringRef qirName = "__quantum__qis__h__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Get the qubit argument from the operation
        Value inputQubit = adaptor.getInput();

        // Create the call operation to apply the Hadamard gate
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{inputQubit});

        // Erase the original QIR_HOp
        rewriter.eraseOp(op);

        return success();
    }
};

struct XOpPattern : public ConvertOpToLLVMPattern<XOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        XOp op,
        XOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__x__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        Value inputQubit = adaptor.getInput();
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{inputQubit});

        return success();
    }
};

struct YOpPattern : public ConvertOpToLLVMPattern<YOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        YOp op,
        YOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__y__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        Value inputQubit = adaptor.getInput();
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{inputQubit});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ZOpPattern : public ConvertOpToLLVMPattern<ZOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        ZOp op,
        ZOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__z__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        Value inputQubit = adaptor.getInput();
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{inputQubit});
        rewriter.eraseOp(op);
        return success();
    }
};

struct CNOTOpPattern : public ConvertOpToLLVMPattern<CNOTOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        CNOTOp op,
        CNOTOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define the QIR function name for the CNOT gate.
        StringRef qirName = "__quantum__qis__cnot__body";

        // Create the LLVM function type: (ptr, ptr) -> void.
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(
            voidType,
            {ptrType, ptrType},
            /*isVarArg=*/false);

        // Ensure the runtime function is declared.
        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Get the control and target qubits from the operation.
        Value control = adaptor.getControl();
        Value target = adaptor.getTarget();

        // Create the call operation to invoke the CNOT runtime function.
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{control, target});

        // Erase the original CNOT op.
        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpType>
struct RotationOpLowering : public ConvertOpToLLVMPattern<OpType> {
    using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        OpType op,
        typename OpType::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = op.getContext();

        Value inputQubit = adaptor.getInput();
        Value angleOperand = adaptor.getAngle();

        Type f64Type = rewriter.getF64Type();
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto fnType = LLVM::LLVMFunctionType::get(
            voidType,
            {f64Type, ptrType},
            /*isVarArg=*/false);

        StringRef qirFunctionName = getQIRFunctionName();
        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirFunctionName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{angleOperand, inputQubit});

        rewriter.eraseOp(op);
        return success();
    }

protected:
    virtual StringRef getQIRFunctionName() const = 0;
};

struct RzOpLowering : public RotationOpLowering<RzOp> {
    using RotationOpLowering<RzOp>::RotationOpLowering;

protected:
    StringRef getQIRFunctionName() const override
    {
        return "__quantum__qis__rz__body";
    }
};

struct RxOpLowering : public RotationOpLowering<RxOp> {
    using RotationOpLowering<RxOp>::RotationOpLowering;

protected:
    StringRef getQIRFunctionName() const override
    {
        return "__quantum__qis__rx__body";
    }
};

struct MeasureOpPattern : public ConvertOpToLLVMPattern<MeasureOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext* ctx = getContext();

        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);

        // Declare __quantum__qis__mz__body function: (ptr, ptr) -> void.
        StringRef measureFnName = "__quantum__qis__mz__body";
        Type measureFnType =
            LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);
        LLVM::LLVMFuncOp measureFnDecl = ensureFunctionDeclaration(
            rewriter,
            op,
            measureFnName,
            measureFnType);

        Value qubit = adaptor.getInput();
        Value resultPtr = adaptor.getResult();

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            measureFnDecl.getSymName(),
            ValueRange{qubit, resultPtr});

        return success();
    }
};

struct SwapOpPattern : public ConvertOpToLLVMPattern<qir::SwapOp> {
    using ConvertOpToLLVMPattern<qir::SwapOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        qir::SwapOp op,
        qir::SwapOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext* ctx = getContext();

        // Create the LLVM function type for the swap function: (ptr, ptr) ->
        // void.
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto fnType =
            LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);
        StringRef swapFuncName = "__quantum__qis__swap__body";
        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, swapFuncName, fnType);

        // Retrieve the two input qubits from the adaptor.
        // Assuming your QIR_SwapOp defines arguments "input1" and "input2".
        Value input1 = adaptor.getLhs();
        Value input2 = adaptor.getRhs();

        // Create the call operation to invoke the runtime swap function.
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{input1, input2});
        return success();
    }
};

struct ResetOpPattern : public ConvertOpToLLVMPattern<ResetOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        ResetOp op,
        ResetOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext* ctx = getContext();

        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);

        // Declare __quantum__qis__reset__body function: (ptr) -> void.
        StringRef qirResetFnName = "__quantum__qis__reset__body";
        Type resetFnType =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);
        LLVM::LLVMFuncOp resetFnDecl = ensureFunctionDeclaration(
            rewriter,
            op,
            qirResetFnName,
            resetFnType);

        Value qubit = adaptor.getInput();

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            resetFnDecl.getSymName(),
            ValueRange{qubit});

        return success();
    }
};

} // namespace

void ConvertQIRToLLVMPass::runOnOperation()
{
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](qir::QubitType type) -> Type {
        return LLVM::LLVMPointerType::get(type.getContext());
    });
    typeConverter.addConversion([](qir::ResultType type) -> Type {
        return LLVM::LLVMPointerType::get(type.getContext());
    });

    // Retrieve the analysis and test the mapping and verify method.
    auto &analysis = getAnalysis<AllocationAnalysis>();
    if (!analysis.verify()) {
        llvm::errs() << "QubitMapping analysis verification failed!\n";
        return signalPassFailure();
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    qir::populateConvertQIRToLLVMPatterns(typeConverter, patterns, analysis);

    target.addIllegalDialect<qir::QIRDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::qir::populateConvertQIRToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    AllocationAnalysis &analysis)
{
    patterns.add<AllocOpPattern, AllocResultOpPattern>(typeConverter, analysis);

    patterns.add<
        HOpPattern,
        SwapOpPattern,
        XOpPattern,
        YOpPattern,
        ZOpPattern,
        RzOpLowering,
        RxOpLowering,
        CNOTOpPattern,
        MeasureOpPattern,
        ReadMeasurementOpPattern,
        ResetOpPattern>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}
