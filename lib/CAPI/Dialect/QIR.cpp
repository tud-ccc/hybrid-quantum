//===- QIR.cpp - C Interface for QIR dialect --------------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/QIR.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRTypes.h"

using namespace mlir;
using namespace mlir::qir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QIR, qir, QIRDialect)

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAQubitType(MlirType type)
{
    return isa<QubitType>(unwrap(type));
}

MlirType mlirQubitTypeGet(MlirContext ctx)
{
    return wrap(QubitType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// ResultType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAResultType(MlirType type)
{
    return isa<ResultType>(unwrap(type));
}

MlirType mlirResultTypeGet(MlirContext ctx)
{
    return wrap(ResultType::get(unwrap(ctx)));
}
