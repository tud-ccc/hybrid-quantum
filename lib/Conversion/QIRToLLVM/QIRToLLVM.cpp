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

struct InitOpPattern : public ConvertOpToLLVMPattern<InitOp> {
    using ConvertOpToLLVMPattern<InitOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        InitOp op,
        InitOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = op.getContext();

        // Create null pointer for initialize call
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);

        // Define QIR initialization function
        StringRef fnName = "__quantum__rt__initialize";
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto fnType = LLVM::LLVMFunctionType::get(
            voidType,
            {ptrType},
            /*isVarArg=*/false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{nullPtr});

        rewriter.eraseOp(op);
        return success();
    }
};

struct SeedOpPattern : public ConvertOpToLLVMPattern<SeedOp> {
    using ConvertOpToLLVMPattern<SeedOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        SeedOp op,
        SeedOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        Value seedArg = adaptor.getSeed();

        // Define RNG seed function
        StringRef fnName = "set_rng_seed";
        Type voidType = LLVM::LLVMVoidType::get(op.getContext());
        auto fnType = LLVM::LLVMFunctionType::get(
            voidType,
            {rewriter.getI64Type()},
            /*isVarArg=*/false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{seedArg});

        rewriter.eraseOp(op);
        return success();
    }
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

template<typename OpTy>
struct COpPattern : public ConvertOpToLLVMPattern<OpTy> {
    /// qirName must be exactly the __quantum__qis__XXX__body symbol for OpTy.
    COpPattern(LLVMTypeConverter &tc, StringRef qirName)
            : ConvertOpToLLVMPattern<OpTy>(tc),
              qirName(qirName)
    {}

    LogicalResult matchAndRewrite(
        OpTy op,
        typename OpTy::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = op.getContext();

        // build the (ptr, ptr) -> void function type
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        auto fnTy = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            ArrayRef<Type>{ptrTy, ptrTy},
            /*vararg=*/false);

        // ensure the runtime declaration
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, fnTy);

        // pull the two operands (control, target)
        auto control = adaptor.getOperands()[0];
        auto target = adaptor.getOperands()[1];

        // replace with a direct call
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{}, // no results
            fnDecl.getSymName(),
            ValueRange{control, target});
        return success();
    }

private:
    StringRef qirName;
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

struct RyOpLowering : public RotationOpLowering<RyOp> {
    using RotationOpLowering<RyOp>::RotationOpLowering;

protected:
    StringRef getQIRFunctionName() const override
    {
        return "__quantum__qis__ry__body";
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

// Add to your existing patterns
struct U3OpLowering : public ConvertOpToLLVMPattern<U3Op> {
    using ConvertOpToLLVMPattern<U3Op>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        U3Op op,
        U3OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = getContext();

        Value qubit = adaptor.getInput();
        Value theta = adaptor.getTheta();
        Value phi = adaptor.getPhi();
        Value lambda = adaptor.getLambda();

        // Decompose U(theta, phi, lambda) = RZ(phi) → RY(theta) → RZ(lambda)
        createRotationGate(
            rewriter,
            loc,
            "__quantum__qis__rz__body",
            phi,
            qubit,
            op);
        createRotationGate(
            rewriter,
            loc,
            "__quantum__qis__ry__body",
            theta,
            qubit,
            op);
        createRotationGate(
            rewriter,
            loc,
            "__quantum__qis__rz__body",
            lambda,
            qubit,
            op);

        rewriter.eraseOp(op);
        return success();
    }

private:
    void createRotationGate(
        ConversionPatternRewriter &rewriter,
        Location loc,
        StringRef qirFnName,
        Value angle,
        Value qubit,
        Operation* op) const
    {
        Type f64Type = rewriter.getF64Type();
        Type ptrType = LLVM::LLVMPointerType::get(getContext());
        Type voidType = LLVM::LLVMVoidType::get(getContext());

        auto fnType = LLVM::LLVMFunctionType::get(
            voidType,
            {f64Type, ptrType},
            /*isVarArg=*/false);

        LLVM::LLVMFuncOp fnDecl =
            ensureFunctionDeclaration(rewriter, op, qirFnName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{angle, qubit});
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

// U1(λ) ≡ Rz(λ)
struct U1OpLowering : public ConvertOpToLLVMPattern<U1Op> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        U1Op op,
        U1OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto lambda = adaptor.getLambda();
        auto qubit = adaptor.getInput();

        // __quantum__qis__u1__body(double, ptr) -> void
        StringRef fnName = "__quantum__qis__u1__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {rewriter.getF64Type(), LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{lambda, qubit});
        rewriter.eraseOp(op);
        return success();
    }
};

// U2(φ, λ)
struct U2OpLowering : public ConvertOpToLLVMPattern<U2Op> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        U2Op op,
        U2OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto phi = adaptor.getPhi();
        auto lambda = adaptor.getLambda();
        auto qubit = adaptor.getInput();

        // __quantum__qis__u2__body(double, double, ptr) -> void
        StringRef fnName = "__quantum__qis__u2__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {rewriter.getF64Type(),
             rewriter.getF64Type(),
             LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{phi, lambda, qubit});
        rewriter.eraseOp(op);
        return success();
    }
};

// Controlled-Rz
struct CRzOpLowering : public ConvertOpToLLVMPattern<CRzOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        CRzOp op,
        CRzOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto angle = adaptor.getAngle();
        auto ctrl = adaptor.getControl();
        auto tgt = adaptor.getTarget();

        StringRef fnName = "__quantum__qis__crz__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {rewriter.getF64Type(),
             LLVM::LLVMPointerType::get(getContext()),
             LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{angle, ctrl, tgt});
        return success();
    }
};

// Controlled-Ry
struct CRyOpLowering : public ConvertOpToLLVMPattern<CRyOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        CRyOp op,
        CRyOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto angle = adaptor.getAngle();
        auto ctrl = adaptor.getControl();
        auto tgt = adaptor.getTarget();

        StringRef fnName = "__quantum__qis__cry__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {rewriter.getF64Type(),
             LLVM::LLVMPointerType::get(getContext()),
             LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{angle, ctrl, tgt});
        return success();
    }
};

// Toffoli (CCX)
struct CCXOpLowering : public ConvertOpToLLVMPattern<CCXOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        CCXOp op,
        CCXOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto c1 = adaptor.getControl1();
        auto c2 = adaptor.getControl2();
        auto tgt = adaptor.getTarget();

        StringRef fnName = "__quantum__qis__ccx__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {LLVM::LLVMPointerType::get(getContext()),
             LLVM::LLVMPointerType::get(getContext()),
             LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{c1, c2, tgt});
        return success();
    }
};

// S gate
struct SOpPattern : public ConvertOpToLLVMPattern<SOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        SOp op,
        SOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto in = adaptor.getInput();
        StringRef fnName = "__quantum__qis__s__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{in});
        return success();
    }
};

// Sdg gate
struct SdgOpPattern : public ConvertOpToLLVMPattern<SdgOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        SdgOp op,
        SdgOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto in = adaptor.getInput();
        StringRef fnName = "__quantum__qis__sdg__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{in});
        return success();
    }
};

// T gate
struct TOpPattern : public ConvertOpToLLVMPattern<TOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        TOp op,
        TOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto in = adaptor.getInput();
        StringRef fnName = "__quantum__qis__t__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{in});
        return success();
    }
};

// Tdg gate
struct TdgOpPattern : public ConvertOpToLLVMPattern<TdgOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(
        TdgOp op,
        TdgOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto in = adaptor.getInput();
        StringRef fnName = "__quantum__qis__tdg__body";
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(getContext()),
            {LLVM::LLVMPointerType::get(getContext())},
            false);
        auto fnDecl = ensureFunctionDeclaration(rewriter, op, fnName, fnType);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange{},
            fnDecl.getSymName(),
            ValueRange{in});
        return success();
    }
};

struct BarrierOpPattern : public ConvertOpToLLVMPattern<qir::BarrierOp> {
    using ConvertOpToLLVMPattern<qir::BarrierOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        qir::BarrierOp op,
        qir::BarrierOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // No runtime effect — just erase the op.
        rewriter.eraseOp(op);
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
        InitOpPattern,
        SeedOpPattern,
        HOpPattern,
        XOpPattern,
        YOpPattern,
        ZOpPattern,
        RzOpLowering,
        RxOpLowering,
        RyOpLowering,
        U1OpLowering,
        U2OpLowering,
        U3OpLowering,
        CRzOpLowering,
        CRyOpLowering,
        CCXOpLowering,
        SOpPattern,
        SdgOpPattern,
        TOpPattern,
        TdgOpPattern,
        BarrierOpPattern,
        MeasureOpPattern,
        ReadMeasurementOpPattern,
        ResetOpPattern>(typeConverter);

    patterns.add<COpPattern<CNOTOp>>(
        typeConverter,
        "__quantum__qis__cnot__body");
    patterns.add<COpPattern<CZOp>>(typeConverter, "__quantum__qis__cz__body");
    patterns.add<COpPattern<SwapOp>>(
        typeConverter,
        "__quantum__qis__swap__body");
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}
