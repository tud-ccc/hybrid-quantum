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

#include <algorithm>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>
#include <optional>

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

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const override
    {
        bool hasApplied = false;
        SmallPtrSet<Operation*, 4> markDelete;
        SmallPtrSet<Operation*, 4> markMove;
        for (auto [thenArg, elseArg] : llvm::zip(
                 op.getThenRegion().getArguments(),
                 op.getElseRegion().getArguments())) {
            DenseMap<Value, Value> valuesMap;
            auto mapValue = [&](Value lhs, Value rhs) {
                auto insertion = valuesMap.insert({lhs, rhs});
                return success(insertion.first->second == rhs);
            };
            for (auto thenOp : thenArg.getUsers()) {
                for (auto elseOp : elseArg.getUsers()) {
                    if (OperationEquivalence::isEquivalentTo(
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
        }
        for (auto moveOp : markMove) {
            SmallVector<Value, 4> operands;
            for (auto operand : moveOp->getOperands())
                if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                    unsigned index = blockArg.getArgNumber();
                    operands.push_back(op->getOperand(index + 1));
                }

            rewriter.replaceAllUsesWith(
                moveOp->getResults(),
                moveOp->getOperands());
            rewriter.moveOpBefore(moveOp, op);
            rewriter.replaceAllUsesWith(operands, moveOp->getResults());
            moveOp->setOperands(operands);
        }
        for (auto delOp : markDelete)
            rewriter.replaceOp(delOp, delOp->getOperands());

        return hasApplied ? success() : failure();
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
