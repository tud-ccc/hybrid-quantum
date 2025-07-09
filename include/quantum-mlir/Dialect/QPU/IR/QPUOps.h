/// Declaration of the QPU dialect ops.
///
/// @file

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.h"

#include "llvm/ADT/STLExtras.h"

#include <llvm/Support/Casting.h>

namespace mlir {
namespace qpu {} // namespace qpu
} // namespace mlir

//===----------------------------------------------------------------------===//
//===- Generated includes
//-------------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h.inc"

//===----------------------------------------------------------------------===//
