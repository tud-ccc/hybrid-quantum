//===- Quantum.cpp - C Interface for Quantum dialect ----------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/Quantum.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::quantum;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quantum, quantum, QuantumDialect)

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAQuantumQubitType(MlirType type)
{
    return isa<QubitType>(unwrap(type));
}

MlirType mlirQuantumQubitTypeGet(MlirContext ctx, int64_t length)
{
    return wrap(QubitType::get(unwrap(ctx), length));
}
