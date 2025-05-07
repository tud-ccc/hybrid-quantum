/// Implements the SCF to RVSDG transformation for Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_SCFTORVSDG
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct ScfToRVSDGPass : mlir::quantum::impl::ScfToRVSDGBase<ScfToRVSDGPass> {
    using ScfToRVSDGBase::ScfToRVSDGBase;

    void runOnOperation() override;
};

struct TransformScfIfOp : public OpConversionPattern<scf::IfOp> {
    using OpConversionPattern<scf::IfOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        scf::IfOp op,
        scf::IfOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        return success();
    }
};

} // namespace

void ScfToRVSDGPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [](QubitType type,
           llvm::SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
            // A qubit<1> does not need conversion
            if (type.isSingleQubit()) return std::nullopt;
            // Convert a qubit<N> to N x qubit<1>
            types = SmallVector<Type>(
                type.getSize(),
                QubitType::get(type.getContext(), 1));
            return success();
        });

    target.addDynamicallyLegalOp<mlir::scf::IfOp>(
        [&](scf::IfOp op) { return converter.isLegal(op->getOperandTypes()); });

    populateScfToRVSDGPatterns(converter, patterns);

    // applyPartialOneToNConversion
    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateScfToRVSDGPatterns(
    TypeConverter converter,
    RewritePatternSet &patterns)
{
    patterns.add<TransformScfIfOp>(converter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createScfToRVSDGPass()
{
    return std::make_unique<ScfToRVSDGPass>();
}