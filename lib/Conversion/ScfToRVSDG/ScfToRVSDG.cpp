/// Implements the SCF to RVSDG transformation for Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/SCFToRVSDG/ScfToRVSDG.h"

#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir-c/Rewrite.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/RegionUtils.h>
#include <vector>

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTSCFTORVSDG
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertScfToRVSDGPass
        : mlir::impl::ConvertScfToRVSDGBase<ConvertScfToRVSDGPass> {
    using ConvertScfToRVSDGBase::ConvertScfToRVSDGBase;

    void runOnOperation() override;
};

void copyIfRegion(
    Region &target,
    Region &region,
    llvm::SetVector<Value> &capturedValues,
    RewriterBase &rewriter)
{
    IRMapping mapping;
    for (auto [in, use] :
         llvm::zip_equal(capturedValues, target.getArguments())) {
        mapping.map(in, use);
    }
    rewriter.setInsertionPointToStart(&target.front());
    region.front().walk([&](Operation* op) {
        if (llvm::isa<scf::YieldOp>(op)) {
            llvm::SetVector<Value> newYields;
            for (auto yieldedValue : op->getOperands())
                newYields.insert(mapping.lookup(yieldedValue));
            for (auto arg : target.getArguments()) newYields.insert(arg);
            rewriter.create<rvsdg::YieldOp>(
                op->getLoc(),
                newYields.takeVector());
        } else {
            auto cloned = op->clone(mapping);
            rewriter.insert(cloned);
        }
        rewriter.eraseOp(op);
        return WalkResult::advance();
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
        auto condition = adaptor.getCondition();
        // TODO: Create utility functions that generate True and False
        std::vector<Attribute> matches;
        // True 1 -> 0
        matches.emplace_back(rvsdg::MatchRuleAttr::get(getContext(), {1}, 0));
        // False 0 -> 1
        matches.emplace_back(rvsdg::MatchRuleAttr::get(getContext(), {0}, 1));
        auto mappings = ArrayAttr::get(getContext(), matches);
        auto predicate = rewriter.create<rvsdg::MatchOp>(
            op->getLoc(),
            rvsdg::ControlType::get(getContext(), 2),
            condition,
            mappings);

        // outputTypes = original outputs + captured values that are changed
        llvm::SmallVector<Type> outputTypes(op.getResultTypes());
        llvm::SmallVector<Value> additionalValues;
        for (auto captured : capturedValues) {
            // We only have to add results when they are not a
            // quantum::QuantumType or already returned
            if (!llvm::isa<quantum::QubitType>(captured.getType())
                || !llvm::is_contained(op.getResults(), captured)) {
                additionalValues.emplace_back(captured);
                outputTypes.emplace_back(captured.getType());
            }
        }

        auto gammaOp = rewriter.create<rvsdg::GammaNode>(
            op->getLoc(),
            outputTypes,
            predicate,
            capturedValues.getArrayRef(),
            2);

        llvm::SmallVector<Type> argumentTypes;
        llvm::SmallVector<Location> argumentLocations;
        for (auto arg : capturedValues) {
            argumentTypes.emplace_back(arg.getType());
            argumentLocations.emplace_back(arg.getLoc());
        }

        // Map if/else regions into gammaOp
        auto &thenRegion = gammaOp.getRegion(0);
        if (!thenRegion.hasOneBlock()) thenRegion.emplaceBlock();
        thenRegion.addArguments(argumentTypes, argumentLocations);
        copyIfRegion(thenRegion, op.getThenRegion(), capturedValues, rewriter);

        auto &elseRegion = gammaOp.getRegion(1);
        if (!elseRegion.hasOneBlock()) elseRegion.emplaceBlock();
        elseRegion.addArguments(argumentTypes, argumentLocations);

        // elseRegion can only be empty if the original scf::If did not return
        // any values. Thus, we only return all captured values.
        if (op.getElseRegion().empty()) {
            rewriter.setInsertionPointToStart(&elseRegion.front());
            llvm::SmallVector<Value> newYields(elseRegion.getArguments());
            rewriter.create<rvsdg::YieldOp>(op->getLoc(), newYields);
        } else {
            copyIfRegion(
                elseRegion,
                op.getElseRegion(),
                capturedValues,
                rewriter);
        }

        size_t numResults = op->getNumResults();
        // Replace all original op result uses with the new results
        for (size_t i = 0; i < numResults; ++i) {
            rewriter.replaceAllUsesWith(
                op->getResult(i),
                gammaOp->getResult(i));
        }
        DominanceInfo dom;
        // Replace all new op results with the captured values
        for (size_t i = 0; i < additionalValues.size(); ++i) {
            rewriter.replaceUsesWithIf(
                additionalValues[i],
                gammaOp->getResult(i + numResults),
                [&](OpOperand &operand) {
                    return operand.getOwner() != op
                           && operand.getOwner()->getParentOp() != op
                           && dom.dominates(op, operand.getOwner());
                });
        }
        rewriter.eraseOp(op);
        return success();
    }
};

template<typename... Ts>
void filterForTypes(
    llvm::SmallPtrSetImpl<Region*> &regions,
    SetVector<Value> &filteredValues)
{
    SetVector<Value> usedValues;
    for (Region* region : regions)
        mlir::getUsedValuesDefinedAbove(*region, *region, usedValues);

    for (auto value : usedValues)
        if (llvm::isa<Ts...>(value.getType())) filteredValues.insert(value);
}

} // namespace

void ConvertScfToRVSDGPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });

    target.addLegalDialect<rvsdg::RVSDGDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    // TODO: In general we could set scf::IfOp illegal and transform *every*
    // scf.if to rvsdg.gamma. For now just do it to those that use QubitType
    target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) {
        SmallPtrSet<Region*, 4> regions;
        regions.insert(&op.getThenRegion());
        regions.insert(&op.getElseRegion());

        SetVector<Value> usedValues;
        filterForTypes<qillr::QubitType, quantum::QubitType>(
            regions,
            usedValues);
        return usedValues.empty();
    });

    populateConvertScfToRVSDGPatterns(converter, patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::rvsdg::populateConvertScfToRVSDGPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<TransformScfIfOp>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertScfToRVSDGPass()
{
    return std::make_unique<ConvertScfToRVSDGPass>();
}
