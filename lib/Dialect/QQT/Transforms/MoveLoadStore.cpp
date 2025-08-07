/// Implements the QQT Load Store movement
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h"

#include <memory>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::qqt;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qqt {

#define GEN_PASS_DEF_LOADSTOREMOVE
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h.inc"

} // namespace mlir::qqt

//===----------------------------------------------------------------------===//

namespace {

struct LoadStoreMovePass
        : mlir::qqt::impl::LoadStoreMoveBase<LoadStoreMovePass> {
    using LoadStoreMoveBase::LoadStoreMoveBase;

    void runOnOperation() override;
};
} // namespace

void LoadStoreMovePass::runOnOperation() { auto context = &getContext(); }

std::unique_ptr<Pass> mlir::qqt::createLoadStoreMovePass()
{
    return std::make_unique<LoadStoreMovePass>();
}
