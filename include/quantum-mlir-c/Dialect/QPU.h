//===---- quantum-mlir-c/Dialect/QPU.h - C API for QPU dialect -----*- C-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_QPU_H
#define QUANTUM_MLIR_C_DIALECT_QPU_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QPU, qpu);

//===---------------------------------------------------------------------===//
// MatchRuleAttr
//===---------------------------------------------------------------------===//

/// Returns `true` if the given attribute is a qpu::TargetAttr dialect
/// attribute.
MLIR_CAPI_EXPORTED bool mlirAttrIsATargetAttr(MlirAttribute attr);

/// Creates an qpu::TargetAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirTargetAttrGet(
    MlirContext ctx,
    MlirAttribute qubits,
    MlirAttribute coupling);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_QPU_H
