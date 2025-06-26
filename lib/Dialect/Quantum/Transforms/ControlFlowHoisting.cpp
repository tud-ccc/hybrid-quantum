/// Implements the gate optimization.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_CONTROLFLOWHOISTING
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct ControlFlowHoistingPass : mlir::quantum::impl::ControlFlowHoistingBase<
                                     ControlFlowHoistingPass> {
    using ControlFlowHoistingBase::ControlFlowHoistingBase;

    void runOnOperation() override;
};

struct HoistOperations : OpRewritePattern<IfOp> {
    using OpRewritePattern<IfOp>::OpRewritePattern;

    void findOperandsToIfArgs(
        IfOp op,
        OperandRange operands,
        IRMapping &mapping) const
    {
        for (auto operand : operands)
            if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                unsigned index = blockArg.getArgNumber();
                // IfOp has condition + block args
                mapping.map(operand, op->getOperand(index + 1));
            } else
                assert(false && "Not a block argument");
    }

    LogicalResult findEquivalentOperations(
        SmallPtrSetImpl<Operation*> &markMove,
        SmallPtrSetImpl<Operation*> &markDelete,
        Region &region1,
        Region &region2) const
    {
        bool hasApplied = false;
        for (auto [thenArg, elseArg] :
             llvm::zip(region1.getArguments(), region2.getArguments())) {
            DenseMap<Value, Value> valuesMap;
            auto mapValue = [&](Value lhs, Value rhs) {
                if (!dyn_cast<BlockArgument>(lhs)
                    || !dyn_cast<BlockArgument>(rhs))
                    return failure();
                auto insertion = valuesMap.insert({lhs, rhs});
                return success(insertion.first->second == rhs);
            };
            for (auto [thenOp, elseOp] :
                 llvm::zip(thenArg.getUsers(), elseArg.getUsers())) {
                if (!llvm::isa<YieldOp>(thenOp) && !llvm::isa<YieldOp>(elseOp)
                    && OperationEquivalence::isEquivalentTo(
                        thenOp,
                        elseOp,
                        mapValue,
                        mapValue,
                        OperationEquivalence::IgnoreLocations)) {
                    markMove.insert(thenOp);
                    markDelete.insert(elseOp);
                    hasApplied = true;
                }
            }
        }
        return success(hasApplied);
    }

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const override
    {
        // Find equivalent operations in both branches whose operands
        // depend on the branch's block arguments
        SmallPtrSet<Operation*, 4> markDelete;
        SmallPtrSet<Operation*, 4> markMove;
        if (failed(findEquivalentOperations(
                markMove,
                markDelete,
                op.getThenRegion(),
                op.getElseRegion())))
            return failure();

        for (auto moveOp : markMove) {
            // Remap cloned operation operands to if operands
            IRMapping mapping;
            findOperandsToIfArgs(op, moveOp->getOperands(), mapping);

            // Clone the operation in front of if
            rewriter.setInsertionPoint(op);
            auto clonedOp = rewriter.clone(*moveOp, mapping);

            // Replace the operation from each branch by replacing its uses
            SmallVector<Value, 4> replacements;
            for (auto operand : moveOp->getOperands())
                if (llvm::isa<QubitType>(operand.getType()))
                    replacements.push_back(operand);
            rewriter.replaceOp(moveOp, replacements);

            // Remap if operands to cloned operation results
            for (auto [arg, res] :
                 llvm::zip(clonedOp->getOperands(), clonedOp->getResults()))
                op->replaceUsesOfWith(arg, res);
        }
        // Replace the operation from each branch by replacing its uses
        for (auto delOp : markDelete) {
            SmallVector<Value, 4> replacements;
            for (auto operand : delOp->getOperands())
                if (llvm::isa<QubitType>(operand.getType()))
                    replacements.push_back(operand);
            rewriter.replaceOp(delOp, replacements);
        }

        return success();
    }
};

} // namespace

void ControlFlowHoistingPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    populateControlFlowHoistingPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
}

void mlir::quantum::populateControlFlowHoistingPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<HoistOperations>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createControlFlowHoistingPass()
{
    return std::make_unique<ControlFlowHoistingPass>();
}
