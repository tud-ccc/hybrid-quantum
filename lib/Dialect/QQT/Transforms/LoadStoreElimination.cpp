/// Implements the QQT Load Store movement
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTTypes.h"
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <map>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/RegionUtils.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qqt;

//===- Generated includes -------------------------------------------------===//

namespace mlir::qqt {

#define GEN_PASS_DEF_LOADSTOREELIMINATION
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h.inc"

} // namespace mlir::qqt

//===----------------------------------------------------------------------===//

namespace {

Operation* findImmediatelyPostDominatingLoad(
    StoreOp storeOp,
    PostDominanceInfo &postDomInfo)
{
    // Collect all loads on the same qubit reference
    // that properly post-dominate this store operation
    llvm::SmallVector<Operation*> postDominatingRefLoads =
        llvm::filter_to_vector(
            storeOp.getReference().getUsers(),
            [&](Operation* refUser) {
                return llvm::dyn_cast<qqt::LoadOp>(refUser)
                       && postDomInfo.properlyPostDominates(refUser, storeOp);
            });

    llvm::SmallVector<Operation*> candidates(postDominatingRefLoads);
    while (candidates.size() > 1) {
        llvm::SmallVector<Operation*> toRemove;
        for (auto loadOp : candidates) {
            for (auto otherLoadOp : postDominatingRefLoads)
                if (postDomInfo.properlyPostDominates(loadOp, otherLoadOp)) {
                    toRemove.emplace_back(loadOp);
                    continue;
                }
        }
        for (auto rem : toRemove) candidates.erase(llvm::find(candidates, rem));
    }
    return candidates.back();
}

Operation* findImmediateDominatingStore(LoadOp loadOp, DominanceInfo &domInfo)
{
    // Collect all stores on the same qubit reference
    // that properly dominate this load operation
    llvm::SmallVector<Operation*> dominatingRefStores = llvm::filter_to_vector(
        loadOp.getInput().getUsers(),
        [&](Operation* refUser) {
            return llvm::dyn_cast<qqt::StoreOp>(refUser)
                   && domInfo.properlyDominates(refUser, loadOp);
        });

    llvm::SmallVector<Operation*> candidates(dominatingRefStores);
    while (candidates.size() > 1) {
        llvm::SetVector<Operation*> toRemove;
        for (auto storeOp : candidates) {
            for (auto otherStoreOp : dominatingRefStores)
                if (storeOp != otherStoreOp
                    && domInfo.properlyDominates(storeOp, otherStoreOp)) {
                    toRemove.insert(storeOp);
                    continue;
                }
        }
        for (auto rem : toRemove) candidates.erase(llvm::find(candidates, rem));
    }
    return candidates.back();
}

struct LoadStoreEliminationPass
        : mlir::qqt::impl::LoadStoreEliminationBase<LoadStoreEliminationPass> {
    using LoadStoreEliminationBase::LoadStoreEliminationBase;

    void runOnOperation() override;
};
} // namespace

// TODO: In the future this operations should run over
// the qpu.circuit as all quantum code will be contained in circuits
void LoadStoreEliminationPass::runOnOperation()
{
    llvm::SetVector<Operation*> erasures;
    auto &domInfo = getAnalysis<DominanceInfo>();
    // Find each load and remove it plus its idom(store)
    // Relink the qubits involed
    getOperation()->walk([&](qqt::LoadOp loadOp) {
        if (auto storeOp = llvm::dyn_cast<qqt::StoreOp>(
                findImmediateDominatingStore(loadOp, domInfo))) {
            loadOp.getResult().replaceAllUsesWith(storeOp.getQubit());
            erasures.insert(loadOp);
            erasures.insert(storeOp);
        }
    });
    // erase all load and store operations found

    for (auto op : erasures) op->erase();

    // Remove all unused !qqt.ref values
    getOperation().walk([](rvsdg::GammaNode gamma) {
        llvm::BitVector bits(gamma->getNumOperands(), false);
        for (auto it : llvm::enumerate(gamma->getOperands()))
            if (llvm::isa<qqt::QubitRefType>(it.value().getType()))
                bits[it.index()] = true;

        llvm::BitVector argumentBits(bits.size() - 1, false);
        for (unsigned i = 0; i < argumentBits.size(); ++i)
            argumentBits[i] = bits[i + 1];

        for (auto &r : gamma->getRegions())
            r.getBlocks().front().eraseArguments(argumentBits);
        gamma->eraseOperands(bits);
    });

    // Remove the unused promote / destruct operations
    getOperation()->walk([](qqt::PromoteOp promote) {
        for (auto use : promote.getResult().getUsers()) use->erase();
        promote.erase();
    });

    return markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

std::unique_ptr<Pass> mlir::qqt::createLoadStoreEliminationPass()
{
    return std::make_unique<LoadStoreEliminationPass>();
}
