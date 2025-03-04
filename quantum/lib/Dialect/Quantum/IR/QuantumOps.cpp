/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

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

// //Verfiers
// LogicalResult XOp::verify() {
//   if (getInput().getType() != getResult().getType())
//     return emitOpError("input and result must have the same type");
//   return success();
// }

// LogicalResult CNOTOp::verify() {
//   return success();
// }

// LogicalResult InsertOp::verify()
// {
//     if (!(getIdx() || getIdxAttr().has_value())) {
//         return emitOpError() << "expected op to have a non-null index";
//     }
//     return success();
// }

// LogicalResult ExtractOp::verify()
// {
//     return success();
// }

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