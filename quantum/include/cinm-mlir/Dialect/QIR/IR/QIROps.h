/// Declaration of the QIR dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/QIR/IR/QIRBase.h"
#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"



//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/QIR/IR/QIROps.h.inc"

//===----------------------------------------------------------------------===//
