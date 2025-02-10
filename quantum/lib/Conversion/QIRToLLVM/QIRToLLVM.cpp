/// Implements the ConvertQIRToLLVMPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QIRToLLVM/QIRToLLVM.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::qir;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQIRTOLLVM
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//


namespace {

struct ConvertQIRToLLVMPass
        : mlir::impl::ConvertQIRToLLVMBase<ConvertQIRToLLVMPass> {
    using ConvertQIRToLLVMBase::ConvertQIRToLLVMBase;

    void runOnOperation() override;
};


LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
};


struct AllocOpPattern : public ConvertOpToLLVMPattern<AllocOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(AllocOp op, AllocOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for single-qubit allocation
        StringRef qirName = "__quantum__rt__qubit_allocate";

        // Create the function type: () -> !llvm.ptr
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(ptrType, {}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Create the call operation to allocate a single qubit
        auto callOp = rewriter.create<LLVM::CallOp>(
            loc,
            fnDecl.getResultTypes(),
            fnDecl.getSymName(),
            ValueRange{}  // No arguments since we allocate a single qubit
        );

        // Replace the original op with the result of the call operation
        rewriter.replaceOp(op, callOp.getResult());
        return success();
    }
};


struct AllocResultOpPattern : public ConvertOpToLLVMPattern<AllocResultOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(AllocResultOp op, AllocResultOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for single-qubit allocation
        StringRef qirName = "__quantum__rt__result_allocate";

        // Create the function type: () -> !llvm.ptr
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(ptrType, {}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Create the call operation to allocate a single qubit
        auto callOp = rewriter.create<LLVM::CallOp>(
            loc,
            fnDecl.getResultTypes(),
            fnDecl.getSymName(),
            ValueRange{}  // No arguments since we allocate a single qubit
        );

        // Replace the original op with the result of the call operation
        rewriter.replaceOp(op, callOp.getResult());
        return success();
    }
};

struct HOpPattern : public ConvertOpToLLVMPattern<HOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(HOp op, HOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for the Hadamard gate
        StringRef qirName = "__quantum__qis__h__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Get the qubit argument from the operation
        Value inputQubit = adaptor.getInput();

        // Create the call operation to apply the Hadamard gate
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), ValueRange{inputQubit});

        // Erase the original QIR_HOp
        rewriter.eraseOp(op);

        return success();
    }
};


struct MeasureOpPattern : public ConvertOpToLLVMPattern<MeasureOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(MeasureOp op, MeasureOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for the measure operation
        StringRef qirName = "__quantum__qis__mz__body";

        // Create the function type: (ptr, ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Get the qubit and result arguments from the operation
        Value inputQubit = adaptor.getInput();
        Value resultPtr = adaptor.getResult();

        // Create the call operation to perform the measurement
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), 
                                      ValueRange{inputQubit, resultPtr});

        // Erase the original QIR_MeasureOp
        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

void ConvertQIRToLLVMPass::runOnOperation()
{
    LLVMTypeConverter typeConverter(&getContext());
    // Add custom conversions for QIR types -> LLVM pointers
    typeConverter.addConversion([&](qir::QubitType type) -> Type {
        return LLVM::LLVMPointerType::get(&getContext());
    });
    typeConverter.addConversion([&](qir::ResultType type) -> Type {
        return LLVM::LLVMPointerType::get(&getContext());
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    qir::populateConvertQIRToLLVMPatterns(typeConverter, patterns);    
    target.addIllegalDialect<qir::QIRDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

void mlir::qir::populateConvertQIRToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<AllocOpPattern, HOpPattern, AllocResultOpPattern, MeasureOpPattern>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}





