//===-- quantum-mlir-c/Dialect/RVSDG.h - C API for RVSDG dialect ---*- C-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_RVSDG_H
#define QUANTUM_MLIR_C_DIALECT_RVSDG_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RVSDG, rvsdg);

//===---------------------------------------------------------------------===//
// ControlType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a rvsdg::ControlType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAControlType(MlirType type);

/// Creates an rvsdg::ControlType type.
MLIR_CAPI_EXPORTED MlirType
mlirControlTypeGet(MlirContext ctx, uint64_t numOptions);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_RVSDG_H
