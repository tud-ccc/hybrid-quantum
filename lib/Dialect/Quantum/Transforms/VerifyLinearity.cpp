/// Implements verification of Linearity property of the the No-Clone Theorem
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/Analysis/LinearUseAnalysis.h"
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h"

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>

using namespace mlir;
using namespace mlir::quantum;
using namespace mlir::quantum::dataflow;

//===- Generated includes -------------------------------------------------===//

namespace mlir::quantum {

#define GEN_PASS_DEF_VERIFYLINEARITY
#include "quantum-mlir/Dialect/Quantum/Transforms/Passes.h.inc"

} // namespace mlir::quantum

//===----------------------------------------------------------------------===//

namespace {

struct VerifyLinearityPass
        : quantum::impl::VerifyLinearityBase<VerifyLinearityPass> {
    using VerifyLinearityBase::VerifyLinearityBase;

    void runOnOperation() override;
};
} // namespace

void VerifyLinearityPass::runOnOperation()
{
    mlir::DataFlowConfig config;
    config.setInterprocedural(false);

    mlir::DataFlowSolver solver(config);

    // DenseForwardDataFlowAnalysis relies on executability information.
    // DeadCodeAnalysis in turn uses constant information for conditions.
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();

    auto* linearity = solver.load<LinearUseAnalysis>();

    Operation* root = getOperation();

    if (failed(solver.initializeAndRun(root))) {
        root->emitError("failed to run linear-use analysis");
        signalPassFailure();
        return;
    }

    if (!linearity->hasViolations()) return;

    for (OpOperand* operand : linearity->getViolations()) {
        Operation* user = operand->getOwner();

        user->emitError() << "qubit operand #" << operand->getOperandNumber()
                          << " has already been consumed";
    }

    signalPassFailure();
}

std::unique_ptr<Pass> mlir::quantum::createVerifyLinearityPass()
{ return std::make_unique<VerifyLinearityPass>(); }
