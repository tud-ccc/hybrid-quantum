/// Implements the SABRE Algorithm for QPU/Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/Analysis/SabreSwapAnalysis.h"
#include "quantum-mlir/Dialect/QPU/IR/QPU.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.h"
#include "quantum-mlir/Dialect/QPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::qpu;
// using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qpu {

#define GEN_PASS_DEF_SABRESWAP
#include "quantum-mlir/Dialect/QPU/Transforms/Passes.h.inc"

} // namespace mlir::qpu

//===----------------------------------------------------------------------===//

namespace {

struct SabreSwapPass : mlir::qpu::impl::SabreSwapBase<SabreSwapPass> {
    using SabreSwapBase::SabreSwapBase;

    void runOnOperation() override;
};

} // namespace

void SabreSwapPass::runOnOperation()
{
    Operation* module = getOperation();
    SabreSwapAnalysis analysis(module);
    OpBuilder builder(&getContext());
}

std::unique_ptr<Pass> mlir::qpu::createSabreSwapPass()
{
    return std::make_unique<SabreSwapPass>();
}
