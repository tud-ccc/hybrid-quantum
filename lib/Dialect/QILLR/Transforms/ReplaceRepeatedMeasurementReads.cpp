/// Implements the QILLR ReadMeasurement elimination pass
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h"

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::qillr;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qillr {

#define GEN_PASS_DEF_REPLACEREPEATEDMEASUREMENTREADS
#include "quantum-mlir/Dialect/QILLR/Transforms/Passes.h.inc"

} // namespace mlir::qillr

//===----------------------------------------------------------------------===//

namespace {

Operation* findImmediateDominatingMeasure(
    ReadMeasurementOp readMeasurement,
    DominanceInfo &domInfo)
{
    Operation* op = readMeasurement.getOperation();
    Block* block = op->getBlock();
    Region* region = block->getParent();

    // Case 1: single-block region -> just scan backwards
    if (region->hasOneBlock()) {
        for (Operation* it = op->getPrevNode(); it; it = it->getPrevNode())
            if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(it))
                if (measure.getResult() == readMeasurement.getInput())
                    return measure;
        return nullptr;
    }

    // Case 2: multiple blocks -> use dominator tree
    for (Operation* it = op->getPrevNode(); it; it = it->getPrevNode())
        if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(it))
            if (measure.getResult() == readMeasurement.getInput())
                return measure;

    auto* node = domInfo.getNode(block);
    while (auto* idom = node->getIDom()) {
        Block* idomBlock = idom->getBlock();
        for (auto &blockOp : llvm::reverse(*idomBlock))
            if (auto measure = llvm::dyn_cast<qillr::MeasureOp>(&blockOp))
                if (measure.getResult() == readMeasurement.getInput())
                    return measure;
        node = idom;
    }

    return nullptr;
}

struct ReplaceRepeatedMeasurementReadsPass
        : mlir::qillr::impl::ReplaceRepeatedMeasurementReadsBase<
              ReplaceRepeatedMeasurementReadsPass> {
    using ReplaceRepeatedMeasurementReadsBase::
        ReplaceRepeatedMeasurementReadsBase;

    void runOnOperation() override;
};

struct ReplaceRepeatedReads
        : public OpRewritePattern<qillr::ReadMeasurementOp> {

    DominanceInfo &domInfo;

    ReplaceRepeatedReads(MLIRContext* context, DominanceInfo &domInfo)
            : OpRewritePattern<qillr::ReadMeasurementOp>(context),
              domInfo(domInfo) {};

    LogicalResult matchAndRewrite(
        qillr::ReadMeasurementOp op,
        PatternRewriter &rewriter) const override
    {
        // %r = ralloc
        // measure(%q, %r)
        // %m1 = read_measurement(%r)
        // %m2 = read_measurement(%r)
        // measure(%q, %r)
        // %m3 = read_measurement(%r)
        // replace uses of %m2 by %m1 and remove %m2 = ...
        // but retain %m3
        auto mutualResultAlloc = op.getInput();
        auto dominatingMeasure = findImmediateDominatingMeasure(op, domInfo);

        // There has been no measure(%q, %r) of %r
        if (dominatingMeasure == nullptr) {
            auto zero = rewriter.create<arith::ConstantIntOp>(
                op->getLoc(),
                0,
                rewriter.getI1Type());
            rewriter.replaceAllUsesWith(op.getMeasurement(), zero);
            rewriter.eraseOp(op);
            return success();
        }

        // Find other read_measurement operations that are
        // immediately domminated by the same measurement
        // operation
        for (auto user : mutualResultAlloc.getUsers()) {
            if (ReadMeasurementOp read =
                    llvm::dyn_cast<qillr::ReadMeasurementOp>(user)) {
                if (dominatingMeasure
                        == findImmediateDominatingMeasure(read, domInfo)
                    && op != read) {
                    auto m1 = read.getResult();
                    auto m2 = op.getResult();
                    m2.replaceAllUsesWith(m1);
                    rewriter.eraseOp(op);
                    return success();
                }
            }
        }
        return failure();
    }
};

} // namespace

void ReplaceRepeatedMeasurementReadsPass::runOnOperation()
{
    auto context = &getContext();
    RewritePatternSet patterns(context);

    auto &domInfo = getAnalysis<DominanceInfo>();

    // Fully qualify the function call
    populateReplaceRepeatedMeasurementReadsPatterns(patterns, domInfo);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();

    return markAnalysesPreserved<DominanceInfo>();
}

void mlir::qillr::populateReplaceRepeatedMeasurementReadsPatterns(
    RewritePatternSet &patterns,
    DominanceInfo &domInfo)
{
    patterns.add<ReplaceRepeatedReads>(patterns.getContext(), domInfo);
}

std::unique_ptr<Pass> mlir::qillr::createReplaceRepeatedMeasurementReadsPass()
{
    return std::make_unique<ReplaceRepeatedMeasurementReadsPass>();
}
