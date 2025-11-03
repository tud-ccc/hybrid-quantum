/// Implements the QQT Load Store movement
///
/// @file
/// @author     Lars Schuetze (lars.schuetze@tu-dresden.de)

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "quantum-mlir/Dialect/QQT/IR/QQTOps.h"
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
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

#define GEN_PASS_DEF_LOADSTOREMOVE
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h.inc"

} // namespace mlir::qqt

//===----------------------------------------------------------------------===//

namespace {

void copyIfRegion(
    Region &target,
    Region &region,
    llvm::SmallVector<Value> &capturedValues,
    llvm::SmallVector<qqt::StoreOp> &extraYieldValues,
    OpBuilder &rewriter)
{
    // Map captured values to their block argument value
    IRMapping mapping;
    for (auto [in, use] :
         llvm::zip_equal(capturedValues, target.getArguments())) {
        mapping.map(in, use);
    }
    rewriter.setInsertionPointToStart(&target.front());
    // Empty scf.IfOp region - only create a yield
    if (region.empty()) {
        llvm::SmallVector<Value> newYields;
        for (auto arg : target.getArguments())
            if (llvm::isa<quantum::QubitType>(arg.getType()))
                newYields.emplace_back(arg);
        rewriter.create<rvsdg::YieldOp>(region.getLoc(), newYields);
    } else {
        region.walk([&](Operation* op) {
            // Rewrite the YieldOp:
            if (llvm::isa<scf::YieldOp>(op)) {
                llvm::SmallVector<Value> newYields;
                for (Value v : op->getOperands())
                    newYields.emplace_back(mapping.lookup(v));
                for (qqt::StoreOp store : extraYieldValues)
                    newYields.emplace_back(mapping.lookup(store.getQubit()));
                rewriter.create<rvsdg::YieldOp>(op->getLoc(), newYields);
            } else {
                rewriter.clone(*op, mapping);
            }
            // op->erase();
        });
    }
}

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
        llvm::SetVector<Operation*> toRemove;
        for (auto loadOp : candidates) {
            for (auto otherLoadOp : postDominatingRefLoads)
                if (postDomInfo.properlyPostDominates(loadOp, otherLoadOp)) {
                    toRemove.insert(loadOp);
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
                if (domInfo.properlyDominates(storeOp, otherStoreOp)) {
                    toRemove.insert(storeOp);
                    continue;
                }
        }
        for (auto rem : toRemove) candidates.erase(llvm::find(candidates, rem));
    }
    return candidates.back();
}

bool collectMovableOperations(
    Region &region,
    DominanceInfo &domInfo,
    PostDominanceInfo &postDomInfo,
    std::vector<std::pair<qqt::LoadOp, Operation*>> &moveTop,
    std::vector<std::pair<qqt::StoreOp, Operation*>> &moveBottom)
{
    bool invalidate = false;
    region.walk([&](Operation* op) {
        if (auto load = llvm::dyn_cast<qqt::LoadOp>(op)) {
            auto storeOp = findImmediateDominatingStore(load, domInfo);
            moveTop.emplace_back(std::make_pair(load, storeOp));
            invalidate = true;
        }
        if (auto store = llvm::dyn_cast<qqt::StoreOp>(op)) {
            auto loadOp = findImmediatelyPostDominatingLoad(store, postDomInfo);
            moveBottom.emplace_back(std::make_pair(store, loadOp));
            invalidate = true;
        }
    });
    return invalidate;
}

struct LoadStoreMovePass
        : mlir::qqt::impl::LoadStoreMoveBase<LoadStoreMovePass> {
    using LoadStoreMoveBase::LoadStoreMoveBase;

    void runOnOperation() override;
};
} // namespace

// TODO: In the future this operations should run over
// the qpu.circuit as all quantum code will be contained in circuits
void LoadStoreMovePass::runOnOperation()
{
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();
    getOperation().walk([&](scf::IfOp branch) {
        std::vector<std::pair<qqt::LoadOp, Operation*>> moveLoadsThenRegion;
        std::vector<std::pair<qqt::StoreOp, Operation*>> moveStoresThenRegion;
        std::vector<std::pair<qqt::LoadOp, Operation*>> moveLoadsElseRegion;
        std::vector<std::pair<qqt::StoreOp, Operation*>> moveStoresElseRegion;

        collectMovableOperations(
            branch.getThenRegion(),
            domInfo,
            postDomInfo,
            moveLoadsThenRegion,
            moveStoresThenRegion);

        if (!branch.getElseRegion().empty())
            collectMovableOperations(
                branch.getElseRegion(),
                domInfo,
                postDomInfo,
                moveLoadsElseRegion,
                moveStoresElseRegion);

        // Move the loads out of the branch if possible.
        // This allows to directly capture the used qubit values.
        // TODO: Merge load ops when loading from the same reference
        for (const auto &x : moveLoadsThenRegion) {
            auto load = x.first;
            auto store = x.second;
            if (load->getParentRegion() != store->getParentRegion())
                load->moveBefore(load->getParentOp());
            else
                load->moveAfter(store);
        }

        llvm::SmallVector<Type> resultTypes(branch->getResultTypes());
        llvm::SmallVector<qqt::StoreOp> moveOutStores;
        for (const auto &x : moveStoresThenRegion) {
            auto store = x.first;
            auto load = x.second;
            if (store->getParentRegion() != load->getParentRegion()) {
                // Store moved out of the branch; update the yield op
                moveOutStores.emplace_back(store);
                resultTypes.emplace_back(store.getQubit().getType());
                store->moveAfter(store->getParentOp());
            } else {
                store->moveBefore(load);
            }
        }

        llvm::SetVector<Value> usedAbove;
        mlir::getUsedValuesDefinedAbove(
            branch.getThenRegion(),
            branch.getThenRegion(),
            usedAbove);
        if (!branch.getElseRegion().empty())
            mlir::getUsedValuesDefinedAbove(
                branch.getElseRegion(),
                branch.getElseRegion(),
                usedAbove);

        mlir::OpBuilder builder(branch);

        auto condition = branch.getCondition();
        // TODO: Create utility functions that generate True and False
        std::vector<Attribute> matches;
        // True 1 -> 0
        matches.emplace_back(rvsdg::MatchRuleAttr::get(&getContext(), {1}, 0));
        // False 0 -> 1
        matches.emplace_back(rvsdg::MatchRuleAttr::get(&getContext(), {0}, 1));
        auto mappings = ArrayAttr::get(&getContext(), matches);
        auto predicate = builder.create<rvsdg::MatchOp>(
            branch.getLoc(),
            rvsdg::ControlType::get(&getContext(), 2),
            condition,
            mappings);

        llvm::SmallVector<Value> capturedValues(
            usedAbove.begin(),
            usedAbove.end());
        rvsdg::GammaNode gamma = builder.create<rvsdg::GammaNode>(
            branch->getLoc(),
            resultTypes,
            predicate.getResult(),
            capturedValues,
            /* regionCount*/ 2);

        llvm::SmallVector<Type> argumentTypes;
        llvm::SmallVector<Location> argumentLocations;
        for (auto arg : capturedValues) {
            argumentTypes.emplace_back(arg.getType());
            argumentLocations.emplace_back(arg.getLoc());
        }
        auto &thenRegion = gamma.getRegion(0);
        if (!thenRegion.hasOneBlock()) thenRegion.emplaceBlock();
        thenRegion.addArguments(argumentTypes, argumentLocations);
        copyIfRegion(
            thenRegion,
            branch.getThenRegion(),
            capturedValues,
            moveOutStores,
            builder);

        auto &elseRegion = gamma.getRegion(1);
        if (!elseRegion.hasOneBlock()) elseRegion.emplaceBlock();
        elseRegion.addArguments(argumentTypes, argumentLocations);
        copyIfRegion(
            elseRegion,
            branch.getElseRegion(),
            capturedValues,
            moveOutStores,
            builder);

        IRMapping newStoreRefs;
        for (auto it : llvm::enumerate(moveOutStores)) {
            auto index = it.index();
            auto value = it.value();
            std::size_t resultIndex = branch->getNumResults() + index;
            newStoreRefs.map(value.getQubit(), gamma->getResult(resultIndex));
        }
        for (qqt::StoreOp movedStore : moveOutStores) {
            auto cloned = movedStore->clone(newStoreRefs);
            builder.setInsertionPointAfter(movedStore);
            builder.insert(cloned);
            movedStore.erase();
        }

        branch->erase();
    });
    domInfo.invalidate();
    postDomInfo.invalidate();
}

std::unique_ptr<Pass> mlir::qqt::createLoadStoreMovePass()
{
    return std::make_unique<LoadStoreMovePass>();
}
