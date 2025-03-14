/// Implements the ConvertQIRToLLVMPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "cinm-mlir/Dialect/QIR/IR/QIROps.h"
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

#include <mlir/IR/Types.h>

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOLLVM
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir
//===----------------------------------------------------------------------===//

struct mlir::qir::QubitMapping {
public:
    explicit QubitMapping(Operation* op)
    {
        int64_t allocOpId = 0;
        int64_t allocResultOpId = 0;

        // Walk through all operations in the module and find AllocOp
        op->walk([&](AllocOp allocOp) { mapping[allocOp] = allocOpId++; });

        // Walk through all operations in the module and find AllocResultOp
        op->walk([&](AllocResultOp allocResultOp) {
            resultMapping[allocResultOp] = allocResultOpId++;
        });

        qubitCount = allocOpId;
        resultCount = allocResultOpId;
    }

    int64_t getQubitId(AllocOp allocOp) const
    {
        auto it = mapping.find(allocOp);
        assert(it != mapping.end() && "AllocOp not found in mapping!");
        return it->second;
    }

    int64_t getResultId(AllocResultOp allocResultOp) const
    {
        auto it = resultMapping.find(allocResultOp);
        assert(
            it != resultMapping.end() && "AllocResultOp not found in mapping!");
        return it->second;
    }

    int64_t getQubitCount() const { return qubitCount; }
    int64_t getResultCount() const { return resultCount; }

private:
    int64_t qubitCount = 0;
    int64_t resultCount = 0;
    llvm::DenseMap<AllocOp, int64_t> mapping;
    llvm::DenseMap<AllocResultOp, int64_t> resultMapping;
};

struct mlir::qir::Analysis {
    QubitMapping mapping;
    explicit Analysis(Operation* op) : mapping(op) {}

    // ensure that the counts are non-zero if there are any allocations
    bool verify() const
    {
        return (mapping.getQubitCount() >= 0)
               && (mapping.getResultCount() >= 0);
    }
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
        const QubitMapping &mapping)
            : ConvertOpToLLVMPattern(typeConverter),
              qubitMapping(mapping)
    {}

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Get the unique ID for this AllocOp from the analysis result
        int64_t qubitId = qubitMapping.getQubitId(op);

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
    const QubitMapping &qubitMapping;
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
        auto callOp = rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            i1Type,
            fnDecl.getSymName(),
            ValueRange{inputResult});

        rewriter.replaceOp(op, callOp);

        return success();
    }
};

struct AllocResultOpPattern : public ConvertOpToLLVMPattern<AllocResultOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    AllocResultOpPattern(
        LLVMTypeConverter &typeConverter,
        const QubitMapping &mapping)
            : ConvertOpToLLVMPattern(typeConverter),
              qubitMapping(mapping)
    {}

    LogicalResult matchAndRewrite(
        AllocResultOp op,
        AllocResultOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Get the unique ID for this AllocResultOp from the analysis result
        int64_t resultId = qubitMapping.getResultId(op);

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
    const QubitMapping &qubitMapping;
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
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{inputQubit});
        rewriter.eraseOp(op);
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
        // Get location and context.
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        // Define common LLVM types.
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);

        // Instead of creating a new constant pointer, use the qubit operand
        // from the measure op.
        Value qubit = adaptor.getInput();

        // For the second argument to record_output, if a null is desired, you
        // can create one. However, if you also want to use the allocated qubit
        // pointer, just reuse it. Here we assume that both operands should be
        // the qubit pointer.
        Value resultPtr =
            adaptor.getResult(); // Or, if the runtime expects a null pointer,
                                 // create one accordingly.

        // Declare the __quantum__qis__mz__body function: (ptr, ptr) -> void.
        StringRef qirMName = "__quantum__qis__mz__body";
        Type mFuncType =
            LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);
        LLVM::LLVMFuncOp mFnDecl =
            ensureFunctionDeclaration(rewriter, op, qirMName, mFuncType);

        // Declare the __quantum__qis__reset__body function: (ptr) -> void.
        StringRef qirResetName = "__quantum__qis__reset__body";
        Type resetFuncType =
            LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);
        LLVM::LLVMFuncOp resetFnDecl = ensureFunctionDeclaration(
            rewriter,
            op,
            qirResetName,
            resetFuncType);

        // Now, use the allocated qubit pointer (i.e. the operand) in all calls.
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            mFnDecl.getSymName(),
            ValueRange{qubit, resultPtr});
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            resetFnDecl.getSymName(),
            ValueRange{qubit});

        // Erase the original measure op.
        rewriter.eraseOp(op);
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
    auto &analysis = getAnalysis<Analysis>();
    if (!analysis.verify()) {
        llvm::errs() << "QubitMapping analysis verification failed!\n";
        signalPassFailure();
        return;
    }
    auto &qubitMapping = analysis.mapping;
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    qir::populateConvertQIRToLLVMPatterns(
        typeConverter,
        patterns,
        qubitMapping);

    target.addIllegalDialect<qir::QIRDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

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
    const QubitMapping &qubitMapping)
{
    patterns.add<AllocOpPattern, AllocResultOpPattern>(
        typeConverter,
        qubitMapping);

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
        ReadMeasurementOpPattern>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}
