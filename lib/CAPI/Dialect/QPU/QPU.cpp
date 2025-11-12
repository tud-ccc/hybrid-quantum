//===- QILLR.cpp - C Interface for QILLR dialect --------------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/QPU.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"

using namespace mlir;
using namespace mlir::qpu;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QPU, qpu, QPUDialect)

//===---------------------------------------------------------------------===//
// MatchRuleAttr
//===---------------------------------------------------------------------===//

/// Returns `true` if the given attribute is a qpu::TargetAttr dialect
/// attribute.
bool mlirAttrIsATargetAttr(MlirAttribute attr)
{
    return isa<TargetAttr>(unwrap(attr));
}

/// Creates an qpu::TargetAttr attribute.
MlirAttribute mlirTargetAttrGet(MlirContext ctx, int64_t qubits, uint64_t index)
{
    return wrap(TargetAttr::get(unwrap(ctx), qubits));
}
