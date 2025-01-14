#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::quantum {

//===- Generated passes ---------------------------------------------------===//
#define GEN_PASS_DEF_QUANTUMOPTIMISEPASS
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct HermitianCancel : public OpRewritePattern<quantum::HOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quantum::HOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();                  
    llvm::outs() << "Transforming quantum.H at location: " << loc << "\n";

    // Check if the operation has the Hermitian trait
    // if (op->hasTrait<mlir::OpTrait::Hermitian>()) {
    //   // Get the next operation in the block
    //   Block *block = op->getBlock();
    //   auto nextOpIt = std::next(Block::iterator(op));

    //   // Check if the next operation is also a Hermitian operation of the same type
    //   if (nextOpIt != block->end() && 
    //       nextOpIt->hasTrait<mlir::quantum::Hermitian>() && 
    //       op->getName() == nextOpIt->getName()) {
    //     // Both operations are identical Hermitian operations, cancel them
    //     rewriter.eraseOp(op);
    //     rewriter.eraseOp(*nextOpIt);
        return success();
  }
};

struct AdjointCancel : public OpRewritePattern<quantum::HOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quantum::HOp op,
                                PatternRewriter &rewriter) const final {
    // Implement adjoint cancellation logic here
    return success();
  }
};

struct CircuitInverseCancel : public OpRewritePattern<quantum::CircuitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quantum::CircuitOp op,
                                PatternRewriter &rewriter) const final {
    // Implement circuit inverse cancellation logic here
    return success();
  }
};

struct QuantumOptimisePass : public quantum::impl::QuantumOptimisePassBase<QuantumOptimisePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<HermitianCancel, AdjointCancel, CircuitInverseCancel>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::quantum