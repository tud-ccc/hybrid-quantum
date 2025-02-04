/// Implements the ConvertQuantumToQIRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/QIR/IR/QIR.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIR
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertQuantumToQIRPass
        : mlir::impl::ConvertQuantumToQIRBase<ConvertQuantumToQIRPass> {
    using ConvertQuantumToQIRBase::ConvertQuantumToQIRBase;

    void runOnOperation() override;
};

class QubitMap {
public:
    QubitMap(MLIRContext *ctx) : ctx(ctx), qubits() {}

    // Map a `quantum.Qubit` to (possible many) `qir.qubit`
    void allocate(Value quantum, ValueRange qir) {
        for (auto qubit : qir) {
            qubits[quantum].push_back(qubit);
        }
    }

    MLIRContext *getContext() const { return ctx; }
private:
    MLIRContext *ctx;
    llvm::DenseMap<Value, std::vector<Value>> qubits;
};

struct ConvertAlloc : public OpRewritePattern<AllocOp> {
    ConvertAlloc(MLIRContext *context)
        : OpRewritePattern<AllocOp>(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(
        AllocOp op,
        //AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        //MLIRContext *ctx = getContext();
        //const TypeConverter *conv = getTypeConverter();
        unsigned size = op.getType().cast<quantum::QubitType>().getSize();
        for(unsigned i = 0; i < size; i++) {
            //auto qubit = rewriter.create<mlir::qir::AllocOp>(op.getLoc(),
                //mlir::qir::QubitType::get(getContext()));
        }
        return success();
    }
}; // struct ConvertAllocOp

} // namespace

void ConvertQuantumToQIRPass::runOnOperation()
{
  TypeConverter typeConverter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  quantum::populateConvertQuantumToQIRPatterns(typeConverter, patterns);

  target.addIllegalDialect<quantum::QuantumDialect>();
  target.addLegalDialect<qir::QIRDialect>();

  if (failed(applyPartialConversion(
          getOperation(),
          target,
          std::move(patterns)))) {
    return signalPassFailure();
  }
}

void mlir::quantum::populateConvertQuantumToQIRPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQIRPass() {
    return std::make_unique<ConvertQuantumToQIRPass>();
}


