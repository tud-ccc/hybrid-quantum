/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include <bits/ranges_base.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LogicalResult.h>
#include <ostream>

#define DEBUG_TYPE "quantum-ops"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===- Verifier -----------------------------------------------------------===//

// LogicalResult QuantumDialect::verifyOperationAttribute(Operation *op,
//                                                      NamedAttribute attr) {
//   if (!llvm::isa<UnitAttr>(attr.getValue()) ||
//       attr.getName() != getContainerModuleAttrName())
//     return success();

//   auto module = dyn_cast<ModuleOp>(op);
//   if (!module)
//     return op->emitError("expected '")
//            << getContainerModuleAttrName() << "' attribute to be attached to
//            '"
//            << ModuleOp::getOperationName() << '\'';
//   return success();
// }

//===- Folders ------------------------------------------------------------===//

OpFoldResult HOp::fold(FoldAdaptor adaptor)
{
    // If the input to this H gate was another H gate, remove both.
    if (auto parent = getOperand().getDefiningOp<HOp>())
        return parent.getOperand();
    return nullptr;
}

OpFoldResult XOp::fold(FoldAdaptor adaptor)
{
    // If the input to this H gate was another H gate, remove both.
    if (auto parent = getOperand().getDefiningOp<XOp>())
        return parent.getOperand();
    return nullptr;
}

template<typename ConcreteType>
LogicalResult NoClone<ConcreteType>::verifyTrait(Operation* op)
{
    for (auto value : op->getOpResults()) {
        if (!llvm::isa<quantum::QubitType>(value.getType())) continue;

        auto uses = value.getUses();
        int i = 0;
        // Check if a qubit is used more than once in the same block
        for (auto it = uses.begin(); it != uses.end(); ++it) {
            llvm::errs() << i++;
            llvm::errs() << it->getOwner()->getName();
            auto parent = it->getOwner()->getParentRegion();
            for (auto jt = std::next(it); jt != uses.end(); ++jt) {
                if (parent == jt->getOwner()->getParentRegion()) {
                    return op->emitOpError()
                           << "result qubit #" << value.getResultNumber()
                           << " used more than once within the same block";
                }
            }
        }
        return success();
    }
}

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
        >();
}