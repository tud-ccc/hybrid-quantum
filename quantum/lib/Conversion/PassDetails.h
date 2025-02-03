///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace quantum {
class QuantumDialect;
} // namespace quantum

namespace qir {
class QIRDialect;
} // namespace qir

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/QuantumPasses.h.inc"
#include "cinm-mlir/Conversion/QIRPasses.h.inc"

//===----------------------------------------------------------------------===//
} // namespace mlir/// Declaration of conversion passes for the QIR dialect.