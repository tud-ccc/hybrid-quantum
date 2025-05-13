//===- QIR.cpp - C Interface for QIR dialect --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/QIR.h"

#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QIR, qir, mlir::qir::QIRDialect)
