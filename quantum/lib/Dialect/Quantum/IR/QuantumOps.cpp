/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-ops"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"

//===----------------------------------------------------------------------===//

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

// Folders
// In QuantumOps.cpp
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
//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
        >();
}