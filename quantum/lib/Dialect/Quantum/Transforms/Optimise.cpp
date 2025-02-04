/// Implements the Optimise.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_QUANTUMOPTIMISE
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct QuantumOptimisePass
        : mlir::quantum::impl::QuantumOptimiseBase<QuantumOptimisePass> {
  using QuantumOptimiseBase::QuantumOptimiseBase;

  void runOnOperation() override;
};

//struct HermitianCancel : RewritePattern<quantum::HOp> {
//   using RewritePattern::RewritePattern;

//   LogicalResult matchAndRewrite(
//     quantum::HOp op,
//     PatternRewriter &rewriter) const final
//   {
//     Location loc = op.getLoc();                  
//     llvm::outs() << "Transforming quantum.H at location: " << loc << "\n";

//     // Check if the operation has the Hermitian trait
//     // if (op->hasTrait<mlir::OpTrait::Hermitian>()) {
//     //   // Get the next operation in the block
//     //   Block *block = op->getBlock();
//     //   auto nextOpIt = std::next(Block::iterator(op));

//     //   // Check if the next operation is also a Hermitian operation of the same type
//     //   if (nextOpIt != block->end() && 
//     //       nextOpIt->hasTrait<mlir::quantum::Hermitian>() && 
//     //       op->getName() == nextOpIt->getName()) {
//     //     // Both operations are identical Hermitian operations, cancel them
//     //     rewriter.eraseOp(op);
//     //     rewriter.eraseOp(*nextOpIt);
//         return success();
//   }
// };

// struct AdjointCancel : public OpRewritePattern<quantum::HOp> {
//   AdjointCancel(MLIRContext *context)
//       : OpRewritePattern<AdjointCancel>(context, /*benefit=*/1) {}

//   LogicalResult matchAndRewrite(
//     quantum::HOp op,
//     PatternRewriter &rewriter) const final
//   {
//     // Implement adjoint cancellation logic here
//     return success();
//   }
// };

} // namespace


void QuantumOptimisePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());


    populateQuantumOptimisePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
      getOperation(),
      std::move(patterns)))) {
      signalPassFailure();
    }
}

void mlir::quantum::populateQuantumOptimisePatterns(RewritePatternSet &patterns)
{
    //patterns.add<
    //  HermitianCancel,
    //  AdjointCancel>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::quantum::createQuantumOptimisePass()
{
    return std::make_unique<QuantumOptimisePass>();
}