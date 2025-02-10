#include <string>
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Conversion/QuantumToLLVM/QuantumToLLVM.h"

using namespace mlir;
using namespace mlir::quantum;

namespace {
//UTILITY TRANSFORMATIONS
//-------------------------------------------------------------------------------
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

struct AllocOpPattern : public OpConversionPattern<AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AllocOp op, AllocOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Get the qubit index from the type
        auto qubitIndex = op.getQubit().getType().getId().getInt();
        
        // Create a constant for the qubit index
        auto indexAttr = rewriter.getI64IntegerAttr(qubitIndex);
        auto indexConstant = rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), indexAttr);
        
        // Replace the AllocOp with the constant
        rewriter.replaceOp(op, indexConstant);
        return success();
    }
};



struct HOpPattern : public OpConversionPattern<HOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(HOp op, HOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        
        // Define the QIR function name
        StringRef qirName = "__quantum__qis__h__body";

        // Create the function type: (%Qubit*) -> void
        Type qubitPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
        Type qirSignature = LLVM::LLVMFunctionType::get(voidType, {qubitPtrType}, false);

        // Ensure the function is declared
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Get the qubit index from the operand (which should now be a constant)
        Value qubitIndex = adaptor.getQubit();
        
        // Create the inttoptr operation
        Value qubitPtr = rewriter.create<LLVM::IntToPtrOp>(loc, qubitPtrType, qubitIndex);

        // Create the call operation
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op,
            TypeRange(),
            fnDecl.getSymName(),
            ValueRange{qubitPtr}
        );

        return success();
    }
};


}


namespace mlir::quantum {
void populateQuantumToLLVMConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<AllocOpPattern>(typeConverter, patterns.getContext());
    patterns.add<HOpPattern>(typeConverter, patterns.getContext());
}

} // namespace quantum
