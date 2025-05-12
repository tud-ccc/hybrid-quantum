/// Implements the SCF to RVSDG transformation for Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
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
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/RegionUtils.h>

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

void moveOpsFromBlock(
    Block* from,
    Block* to,
    IRMapping mapping,
    ConversionPatternRewriter &rewriter)
{
    rewriter.setInsertionPointToStart(to);
    from->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
        if (auto yield = llvm::dyn_cast<scf::YieldOp>(innerOp)) {
            // By definition YieldOp is last operation in the block
            auto &lastOp = to->getOperations().back();
            auto quantumYield = llvm::dyn_cast<quantum::YieldOp>(lastOp);
            assert(
                quantumYield && "Last operation of the block was no YieldOp");
            quantumYield->setOperands(yield->getOperands());
            rewriter.eraseOp(innerOp);
        } else
            rewriter.moveOpBefore(
                innerOp,
                rewriter.getBlock(),
                rewriter.getInsertionPoint());

        // Rewrite SSA values from outside to captured
        for (auto [i, v] : llvm::enumerate(innerOp->getOperands()))
            if (auto mapped = mapping.lookupOrNull(v))
                innerOp->setOperand(i, mapped);
    });
}

struct TransformScfIfOp : public OpConversionPattern<scf::IfOp> {
    using OpConversionPattern<scf::IfOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        scf::IfOp op,
        scf::IfOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SetVector<Value> capturedValues;
        mlir::getUsedValuesDefinedAbove(
            op.getThenRegion(),
            op.getThenRegion(),
            capturedValues);
        if (op.elseBlock()) {
            mlir::getUsedValuesDefinedAbove(
                op.getElseRegion(),
                op.getElseRegion(),
                capturedValues);
        }

        auto capturedValueList =
            SmallVector<Value>(capturedValues.begin(), capturedValues.end());

        auto genOp = rewriter.create<quantum::IfOp>(
            op.getLoc(),
            adaptor.getCondition(),
            capturedValueList,
            buildTerminatedBody,
            buildTerminatedBody);

        IRMapping mapping;
        for (auto [arg, captured] : llvm::zip_equal(
                 genOp.getThenRegion().getArguments(),
                 capturedValueList))
            mapping.map(captured, arg);

        // build then region
        moveOpsFromBlock(op.thenBlock(), genOp.thenBlock(), mapping, rewriter);

        // build else region
        if (!op.getElseRegion().empty()) {
            mapping.clear();
            for (auto [arg, captured] : llvm::zip_equal(
                     genOp.getElseRegion().getArguments(),
                     capturedValueList))
                mapping.map(captured, arg);

            moveOpsFromBlock(
                op.elseBlock(),
                genOp.elseBlock(),
                mapping,
                rewriter);
        }

        for (unsigned int i = 0; i < op->getNumResults(); ++i)
            op->getResult(i).replaceAllUsesWith(genOp->getResult(i));
        rewriter.eraseOp(op);
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

    target.addLegalDialect<quantum::QuantumDialect>();
    // Only allow scf::IfOp with classic operands
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
        bool isLegal = true;
        mlir::visitUsedValuesDefinedAbove(
            op.getThenRegion(),
            op.getThenRegion(),
            [&](OpOperand* operand) {
                if (llvm::isa<quantum::QubitType>(operand->get().getType()))
                    isLegal = false;
            });

        if (op.elseBlock()) {
            mlir::visitUsedValuesDefinedAbove(
                op.getThenRegion(),
                op.getThenRegion(),
                [&](OpOperand* operand) {
                    if (llvm::isa<quantum::QubitType>(operand->get().getType()))
                        isLegal = false;
                });
        }
        return isLegal;
    });
    // Only allow scf::YieldOp with classic operands
    target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp op) {
        for (auto result : op.getOperands())
            if (llvm::isa<quantum::QubitType>(result.getType())) return false;

        return true;
    });

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