//===- QILLR.cpp - C Interface for QILLR dialect --------------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/QILLR.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.h"

using namespace mlir;
using namespace mlir::qillr;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QILLR, qillr, QILLRDialect)

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
