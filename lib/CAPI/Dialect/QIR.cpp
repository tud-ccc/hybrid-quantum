//===- QIR.cpp - C Interface for QIR dialect --------------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/QIR.h"

#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRTypes.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QIR, qir, mlir::qir::QIRDialect)

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAQubitType(MlirType type)
{
    return isa<qir::QubitType>(unwrap(type));
}
