#include "cinm-mlir/Conversion/QuantumToLLVM/QuantumToLLVM.h"
#include "cinm-mlir/Conversion/QuantumPasses.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir
{
#define GEN_PASS_DEF_CONVERTQUANTUMTOLLVMPASS
#include "cinm-mlir/Conversion/QuantumPasses.h.inc"
} // namespace mlir

namespace mlir::quantum
{
namespace {
constexpr int32_t NO_POSTSELECT = -1;

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
  }

struct qAllocOp : public OpConversionPattern<AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AllocOp op, AllocOpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        // Get the size from the result type
        auto resultType = op.getResult().getType().cast<qubitType>();
        int64_t size = resultType.getSize();

        // Create constant for size
        Value nQubits = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(size));

        StringRef qirName = "__quantum__rt__qubit_allocate_array";
        Type qirSignature = LLVM::LLVMFunctionType::get(conv->convertType(qubitType::get(ctx, -1)),
                                                        IntegerType::get(ctx, 32));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        if (!fnDecl) {
            (void)rewriter.notifyMatchFailure(op, "Failed to create or find function declaration.");
            return failure();
        }

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, nQubits);
        return success();
    }
};


struct qMeasureOp : public OpConversionPattern<MeasureOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op, 
        MeasureOpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter
    ) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *converter = getTypeConverter();

        // QIR measurement function name
        StringRef qirName = "__quantum__qis__mz__body";

        // Convert qubit type to LLVM pointer type
        Type qubitPtrType = LLVM::LLVMPointerType::get(ctx);
        Type resultType = IntegerType::get(ctx, 1);

        // Prepare function signature
        SmallVector<Type> argTypes = {qubitPtrType};
        Type qirFuncType = LLVM::LLVMFunctionType::get(resultType, argTypes);

        // Ensure function declaration
        auto fnDecl = ensureFunctionDeclaration(
            rewriter, op, qirName, qirFuncType
        );

        // Call the measurement function
        SmallVector<Value> args = {adaptor.getQinp()};
        Value measureResult = rewriter.create<LLVM::CallOp>(loc, fnDecl, args).getResult();
        // Replace the original operation
        rewriter.replaceOp(op, {measureResult, adaptor.getQinp()});

        return success();
    }
};



struct qDeallocOp : public OpConversionPattern<DeallocateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DeallocateOp op, DeallocateOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__rt__qubit_release_array";
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                        conv->convertType(qubitType::get(ctx, -1)));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, adaptor.getOperands());

        return success();
    }
};

struct qExtractOp : public OpConversionPattern<ExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ExtractOp op, 
        ExtractOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override 
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        // QIR runtime function name for qubit extraction
        StringRef qirName = "__quantum__rt__qubit_extract";
        
        // Define QIR function signature 
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(op.getResult().getType()),  // Return type
            {
                conv->convertType(op.getQreg().getType()),  // Input qubit array type
                IntegerType::get(ctx, 32)  // Index type
            }
        );

        // Ensure function declaration exists
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        if (!fnDecl) {
            return rewriter.notifyMatchFailure(op, "Failed to create QIR function declaration");
        }

        // Get input qubit register and index
        Value qreg = adaptor.getQreg();
        Value index = op.getIdx().empty() ? 
            rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32), rewriter.getI32IntegerAttr(0)) : 
            adaptor.getIdx()[0];

        // Replace with QIR function call
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, 
            fnDecl, 
            ArrayRef<Value>{qreg, index}
        );

        return success();
    }
};



struct qInsertOp : public OpConversionPattern<InsertOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(InsertOp op, InsertOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Unravel use-def chain of quantum register values, converting back to reference semantics.
        rewriter.replaceOp(op, adaptor.getQreg());
        return success();
    }
};

// Base class for quantum gate operations
template <typename OpType>
struct qGateOp : public OpConversionPattern<OpType> {
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = this->getTypeConverter();

        // Define QIR function name based on the operation type
        StringRef qirName;
        if constexpr (std::is_same_v<OpType, HOp>) {
            qirName = "__quantum__qis__h";
        } else if constexpr (std::is_same_v<OpType, XOp>) {
            qirName = "__quantum__qis__x";
        } else if constexpr (std::is_same_v<OpType, YOp>) {
            qirName = "__quantum__qis__y";
        } else if constexpr (std::is_same_v<OpType, ZOp>) {
            qirName = "__quantum__qis__z";
        } else if constexpr (std::is_same_v<OpType, SOp>) {
            qirName = "__quantum__qis__s";
        } else if constexpr (std::is_same_v<OpType, TOp>) {
            qirName = "__quantum__qis__t";
        } else if constexpr (std::is_same_v<OpType, SDaggerOp>) {
            qirName = "__quantum__qis__sdg";
        } else if constexpr (std::is_same_v<OpType, TDaggerOp>) {
            qirName = "__quantum__qis__tdg";
        } else if constexpr (std::is_same_v<OpType, CNOTOp>) {
            qirName = "__quantum__qis__cnot";
        } else if constexpr (std::is_same_v<OpType, ROp>) {
            qirName = "__quantum__qis__rx_body";
        } else if constexpr (std::is_same_v<OpType, CYOp>) {
            qirName = "__quantum__qis__cy";
        } else if constexpr (std::is_same_v<OpType, CZOp>) {
            qirName = "__quantum__qis__cz";
        } else if constexpr (std::is_same_v<OpType, SWAPOp>) {
            qirName = "__quantum__qis__swap";
        } else if constexpr (std::is_same_v<OpType, CCXOp>) {
            qirName = "__quantum__qis__ccx";
        } else if constexpr (std::is_same_v<OpType, CCZOp>) {
            qirName = "__quantum__qis__ccz";
        } else if constexpr (std::is_same_v<OpType, CSWAPOp>) {
            qirName = "__quantum__qis__cswap";
        } else if constexpr (std::is_same_v<OpType, UOp>) {
            qirName = "__quantum__qis__u"; 
        } else {
            return rewriter.notifyMatchFailure(op, "Unsupported operation type.");
        }


        // Define the QIR function signature
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                        conv->convertType(qubitType::get(ctx, -1)));

        // Ensure function declaration exists
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        if (!fnDecl) {
            (void)rewriter.notifyMatchFailure(op, "Failed to create or find function declaration.");
            return failure();
        }

        // Create a call to the QIR function
        Value qubit = adaptor.getInput(); // Ensure this method exists
        rewriter.create<LLVM::CallOp>(loc, TypeRange(), fnDecl.getSymName(), ValueRange{qubit});
        // Erase original operation
        rewriter.eraseOp(op);

        return success();
    }
};


// Specific conversion patterns for each gate
struct qHOp : public qGateOp<HOp> { using qGateOp<HOp>::qGateOp; };
struct qXOp : public qGateOp<XOp> { using qGateOp<XOp>::qGateOp; };
struct qYOp : public qGateOp<YOp> { using qGateOp<YOp>::qGateOp; };
struct qZOp : public qGateOp<ZOp> { using qGateOp<ZOp>::qGateOp; };
struct qSOp : public qGateOp<SOp> { using qGateOp<SOp>::qGateOp; };
struct qTOp : public qGateOp<TOp> { using qGateOp<TOp>::qGateOp; };
struct qROp : public qGateOp<ROp> { using qGateOp<ROp>::qGateOp; };
struct qSDaggerOp : public qGateOp<SDaggerOp> { using qGateOp<SDaggerOp>::qGateOp; };
struct qTDaggerOp : public qGateOp<TDaggerOp> { using qGateOp<TDaggerOp>::qGateOp; };
struct qCNOTOp    : public qGateOp<CNOTOp>    { using qGateOp<CNOTOp>::qGateOp; };
struct qUOp    : public qGateOp<UOp>    { using qGateOp<UOp>::qGateOp; };

} // namespace

struct QIRTypeConverter : public LLVMTypeConverter {
  QIRTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([&](qubitType type) { 
      return convertQubitType(type); 
    });
    addConversion([&](qubitType type) { 
      return convertQubitArrayType(type); 
    });
    addConversion([&](observableType type)  { return convertObservableType(type); });
    addConversion([&](resultType type)      { return convertResultType(type); });
  }

   private:
    Type convertQubitType([[maybe_unused]] Type mlirType) 
    {
        return LLVM::LLVMPointerType::get(&getContext());
    }

    Type convertQubitArrayType([[maybe_unused]] Type mlirType) 
    {
        return LLVM::LLVMPointerType::get(&getContext());
    }

    Type convertObservableType([[maybe_unused]] Type mlirType) 
    {
        return this->convertType(IntegerType::get(&getContext(), 32));
    }

    Type convertResultType([[maybe_unused]] Type mlirType) 
    {
       return LLVM::LLVMPointerType::get(&getContext());
    }
};

void populateQuantumToLLVMConversionPatterns(QIRTypeConverter &typeConverter,
                                             RewritePatternSet &patterns) {
    // Use insert to add all conversion patterns at once
    patterns.insert <qAllocOp, qDeallocOp, qExtractOp, qMeasureOp, qInsertOp, qCNOTOp, qHOp, qXOp, qYOp, qZOp, qROp, qUOp> (typeConverter, patterns.getContext());
}

struct ConvertQuantumToLLVMPass
      : public impl::ConvertQuantumToLLVMPassBase<ConvertQuantumToLLVMPass>
  {
    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        // Insert opaque type definitions
        auto module = getOperation();
        insertOpaqueTypeDefinitions(module);
        QIRTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        populateQuantumToLLVMConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        target.addIllegalDialect<quantum::QuantumDialect>();
    

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }    
    }
    
    private:
  void insertOpaqueTypeDefinitions(ModuleOp module) {
    OpBuilder builder(module.getBody(), module.getBody()->begin());
    
    auto insertOpaqueDef = [&](StringRef name) {
      builder.create<LLVM::GlobalOp>(
        module.getLoc(),
        LLVM::LLVMStructType::getOpaque(name, module.getContext()),
        /*isConstant=*/true,
        LLVM::Linkage::External,
        name,
        /*initializer=*/nullptr);
    };

    insertOpaqueDef("Qubit");
    insertOpaqueDef("Result");
    insertOpaqueDef("Array");
  }
  };

 std::unique_ptr<Pass> createConvertQuantumToLLVMPass()
  {
    return std::make_unique<ConvertQuantumToLLVMPass>();
  }

} // namespace mlir::quantum
