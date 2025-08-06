/// InlinerExtension.cpp - QILLR Inliner Extension.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/Extensions/InlinerExtension.h"

#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace mlir;
using namespace mlir::qillr;

//===----------------------------------------------------------------------===//
// QILLRDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with func operations.
struct QILLRInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// Call operations can always be inlined
    bool isLegalToInline(
        Operation* call,
        Operation* callable,
        bool wouldBeCloned) const final
    {
        auto callOp = llvm::dyn_cast<qillr::GateCallOp>(call);
        auto gateOp = llvm::dyn_cast<qillr::GateOp>(callable);

        return callOp && gateOp;
    }

    /// All operations can be inlined.
    bool isLegalToInline(Operation*, Region*, bool, IRMapping &) const final
    {
        return true;
    }

    /// All gate bodies can be inlined.
    bool isLegalToInline(Region*, Region*, bool, IRMapping &) const final
    {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator by replacing it with a new operation
    /// as necessary.
    void handleTerminator(Operation* op, Block* newDest) const final
    {
        auto returnOp = llvm::dyn_cast<qillr::ReturnOp>(op);
        if (!returnOp) return;

        op->erase();
    };

    /// Handle the given inlined terminator by replacing it with a new operation
    /// as necessary.
    void handleTerminator(Operation* op, ValueRange valuesToRepl) const final
    {
        auto returnOp = llvm::dyn_cast<qillr::ReturnOp>(op);
        if (!returnOp) return;
    };
};
} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::qillr::registerInlinerExtension(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext* ctx, qillr::QILLRDialect* dialect) {
        dialect->addInterfaces<QILLRInlinerInterface>();

        // We do not rely on any other dialect to do the inlining
        // No need to call ctx->getOrLoadDialect<...>();
    });
}
