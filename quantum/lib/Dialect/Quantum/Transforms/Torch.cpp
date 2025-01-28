#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::quantum {

//===- Generated passes ---------------------------------------------------===//
#define GEN_PASS_DEF_QUANTUMTORCHPASS
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct QuantumMatMul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    // llvm::outs() << "Transforming linalg.matmul at location: " << loc << "\n";

    // // Get input matrices
    // Value lhs = op.getInputs()[0];
    // Value rhs = op.getInputs()[1];

    // // Transform tensor to a quantum  state
    // auto quantumState1 = rewriter.create<quantum::StatePreparationOp>(loc, lhs);
    // auto quantumState2 = rewriter.create<quantum::StatePreparationOp>(loc, rhs);

    // //MatMul on the 2 quantum states
    // auto quantumMatMulResult = rewriter.create<quantum::MatMulOp>(loc, quantumState1, quantumState2);

    // //Return the quantum result to tensor. 
    // //auto outputTensor = rewriter.create<quantum::QubitToTensor>(loc, quantumMatMulResult);
    // rewriter.replaceOp(op, quantumMatMulResult);
    return success();
  }
};

// struct QuantumMatMul : public OpRewritePattern<linalg::GenericOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
//                                 PatternRewriter &rewriter) const final {
//     // Check if the next operation is linalg.matmul
//     auto nextOp = genericOp.getNextNode();
//     if (!nextOp || !llvm::isa<linalg::MatmulOp>(nextOp)) {
//       return failure(); // Only match if followed by linalg.matmul
//     }

//     // Extract inputs from linalg.generic and linalg.matmul
//     Value lhs = nextOp.getInputs()[0]; // Assuming it's the first input
//     Value rhs = nextOp.getInputs()[1]; // Assuming it's the second input

//     // Create quantum.matmul operation
//     Location loc = genericOp.getLoc();
//     auto quantumResult = rewriter.create<quantum::MatMulOp>(loc, lhs, rhs);

//     // Replace both operations with quantum.matmul
//     rewriter.eraseOp(genericOp);
//     rewriter.replaceOp(nextOp, quantumResult);

//     return success();
//   }
// };

struct QuantumTorchPass : public impl::QuantumTorchPassBase<QuantumTorchPass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantumMatMul>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::quantum