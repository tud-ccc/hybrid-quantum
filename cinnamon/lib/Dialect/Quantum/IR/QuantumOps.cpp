/// Implements the Quantum dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-ops"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
      >();
}

LogicalResult QuantumDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  if (!llvm::isa<UnitAttr>(attr.getValue()) ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';
  return success();
}


//Verfiers
LogicalResult XOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same type");
  return success();
}

LogicalResult CNOTOp::verify() {
  return success();
}

LogicalResult InsertOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null index";
    }
    return success();
}

LogicalResult ExtractOp::verify()
{
    return success();
}


#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"