//===-- quantum-mlir-c/Dialect/QIR.h - C API for QIR dialect ----------*- C
//-*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_QIR_H
#define QUANTUM_MLIR_C_DIALECT_QIR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QIR, qir);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_QIR_H
