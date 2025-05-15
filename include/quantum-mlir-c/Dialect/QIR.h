//===-- quantum-mlir-c/Dialect/QIR.h - C API for QIR dialect ----------*- C
//-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_QIR_H
#define QUANTUM_MLIR_C_DIALECT_QIR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QIR, qir);

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a quantization dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAQubitType(MlirType type);

/// Creates an qir.QubitType type.
MLIR_CAPI_EXPORTED MlirType mlirQubitTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_QIR_H
