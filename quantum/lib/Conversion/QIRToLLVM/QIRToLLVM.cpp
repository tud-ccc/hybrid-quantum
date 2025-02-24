/// Implements the ConvertQIRToLLVMPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
/// @author     Washim Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)


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
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Generate a unique integer ID for the qubit
        static int64_t nextQubitId = 0;
        int64_t qubitId = nextQubitId++;

        // Create an LLVM constant integer to represent the qubit ID
        Type i64Type = rewriter.getI64Type();
        Value intValue = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(qubitId));

        // Create a pointer type
        Type ptrType = LLVM::LLVMPointerType::get(ctx);

        // Create the inttoptr operation
        Value ptrValue = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, intValue);

        // Replace the original op with the inttoptr operation
        rewriter.replaceOp(op, ptrValue);
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

        // Generate a unique integer ID for the qubit
        static int64_t nextResultId = 0;
        int64_t resultId = nextResultId++;

        // Create an LLVM constant integer to represent the qubit ID
        Type i64Type = rewriter.getI64Type();
        Value intValue = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(resultId));

        // Create a pointer type
        Type ptrType = LLVM::LLVMPointerType::get(ctx);

        // Create the inttoptr operation
        Value ptrValue = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, intValue);

        // Replace the original op with the inttoptr operation
        rewriter.replaceOp(op, ptrValue);
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


struct XOpPattern : public ConvertOpToLLVMPattern<XOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(XOp op, XOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__x__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        Value inputQubit = adaptor.getInput();
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), ValueRange{inputQubit});
        rewriter.eraseOp(op);
        return success();
    }
};

struct YOpPattern : public ConvertOpToLLVMPattern<YOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(YOp op, YOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__y__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        Value inputQubit = adaptor.getInput();
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), ValueRange{inputQubit});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ZOpPattern : public ConvertOpToLLVMPattern<ZOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(ZOp op, ZOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the location and context
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        // Define the QIR function name for the X gate
        StringRef qirName = "__quantum__qis__z__body";

        // Create the function type: (ptr) -> void
        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        Value inputQubit = adaptor.getInput();
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), ValueRange{inputQubit});
        rewriter.eraseOp(op);
        return success();
    }
};

struct RzOpLowering : public ConvertOpToLLVMPattern<RzOp> {
    using ConvertOpToLLVMPattern<RzOp>::ConvertOpToLLVMPattern;
  
    LogicalResult matchAndRewrite(RzOp op, RzOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      MLIRContext *ctx = op.getContext();
  
      // Retrieve the qubit operand.
      Value inputQubit = adaptor.getInput();
  
      // Retrieve the angle operand (now declared as F64).
      Value angleOperand = adaptor.getAngle();
      // Optionally, check that it is a constant op.
      auto constantOp = angleOperand.getDefiningOp<LLVM::ConstantOp>();
      if (!constantOp)
        return failure();
      // Verify that the constant is indeed a FloatAttr.
      auto f64Attr = constantOp.getValue().dyn_cast<FloatAttr>();
      if (!f64Attr)
        return failure();
      // (You can also retrieve the double value if needed.)
      double angleValue = f64Attr.getValueAsDouble();
  
      // Build the LLVM function type: (double, %Qubit*) -> void.
      Type f64Type = rewriter.getF64Type();
      Type ptrType = LLVM::LLVMPointerType::get(ctx);
      Type voidType = LLVM::LLVMVoidType::get(ctx);
      auto fnType = LLVM::LLVMFunctionType::get(voidType, {f64Type, ptrType}, /*isVarArg=*/false);
  
      // Get the QIR function declaration for Rz.
      StringRef qirFunctionName = "__quantum__qis__rz__body";
      LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirFunctionName, fnType);
  
      // Create the call operation using the f64 angle operand and qubit operand.
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(),
                                    ValueRange{angleOperand, inputQubit});
  
      // Erase the original op.
      rewriter.eraseOp(op);
      return success();
    }
  };
  
  struct ShowStateOpLowering : public ConvertOpToLLVMPattern<ShowStateOp> {
    using ConvertOpToLLVMPattern<ShowStateOp>::ConvertOpToLLVMPattern;
  
    LogicalResult matchAndRewrite(ShowStateOp op, ShowStateOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      MLIRContext *ctx = op.getContext();
  
      // Create an i64 constant zero.
      Type i64Type = rewriter.getI64Type();
      Value zero = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));
  
      // Create a null pointer of type i8*.
      // Using a zero constant for a pointer type works as a null pointer.
      Type i8PtrType = LLVM::LLVMPointerType::get(ctx);
      Value nullPtr = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrType, zero);

      // Build the function type: (i64, i8*) -> void.
      Type voidType = LLVM::LLVMVoidType::get(ctx);
      auto fnType = LLVM::LLVMFunctionType::get(voidType, {i64Type, i8PtrType}, /*isVarArg=*/false);
  
      // Ensure the function declaration exists.
      StringRef qirFunctionName = "__quantum__rt__tuple_record_output";
      LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirFunctionName, fnType);
  
      // Create the call operation.
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnDecl.getSymName(), ValueRange{zero, nullPtr});
  
      // Erase the original op.
      rewriter.eraseOp(op);
      return success();
    }
  };
  
  


struct MeasureOpPattern : public ConvertOpToLLVMPattern<MeasureOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
    
    LogicalResult matchAndRewrite(MeasureOp op, MeasureOpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override {
      // Get location and context.
      Location loc = op.getLoc();
      MLIRContext *ctx = getContext();
    
      // Define common LLVM types.
      Type ptrType = LLVM::LLVMPointerType::get(ctx);
      Type voidType = LLVM::LLVMVoidType::get(ctx);
  
      // Instead of creating a new constant pointer, use the qubit operand from the measure op.
      Value qubit = adaptor.getInput();
  
      // For the second argument to record_output, if a null is desired, you can create one.
      // However, if you also want to use the allocated qubit pointer, just reuse it.
      // Here we assume that both operands should be the qubit pointer.
      Value resultPtr = adaptor.getResult();  // Or, if the runtime expects a null pointer, create one accordingly.
  
      // Declare the __quantum__qis__mz__body function: (ptr, ptr) -> void.
      StringRef qirMName = "__quantum__qis__mz__body";
      Type mFuncType = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);
      LLVM::LLVMFuncOp mFnDecl = ensureFunctionDeclaration(rewriter, op, qirMName, mFuncType);
    
      // Declare the __quantum__qis__reset__body function: (ptr) -> void.
      StringRef qirResetName = "__quantum__qis__reset__body";
      Type resetFuncType = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);
      LLVM::LLVMFuncOp resetFnDecl = ensureFunctionDeclaration(rewriter, op, qirResetName, resetFuncType);
    
      // Declare the __quantum__rt__result_record_output function: (ptr, i8*) -> void.
      StringRef qirRecordName = "__quantum__rt__result_record_output";
      // We use the same opaque pointer type for i8*.
      Type i8PtrType = LLVM::LLVMPointerType::get(ctx);
      Type recordFuncType = LLVM::LLVMFunctionType::get(voidType, {ptrType, i8PtrType}, false);
      LLVM::LLVMFuncOp recordFnDecl = ensureFunctionDeclaration(rewriter, op, qirRecordName, recordFuncType);
    
      // Now, use the allocated qubit pointer (i.e. the operand) in all calls.
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{}, mFnDecl.getSymName(), ValueRange{qubit, resultPtr});
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{}, resetFnDecl.getSymName(), ValueRange{qubit});
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{}, recordFnDecl.getSymName(), ValueRange{resultPtr, resultPtr});
    
      // Erase the original measure op.
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
    patterns.add<AllocOpPattern, HOpPattern, XOpPattern ,YOpPattern, ZOpPattern,  AllocResultOpPattern, RzOpLowering, ShowStateOpLowering, MeasureOpPattern>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertQIRToLLVMPass()
{
    return std::make_unique<ConvertQIRToLLVMPass>();
}





