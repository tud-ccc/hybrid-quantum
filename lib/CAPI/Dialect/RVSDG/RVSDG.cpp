//===- RVSDG.cpp - C Interface for RVSDG dialect --------------------------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir-c/Dialect/RVSDG.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

using namespace mlir;
using namespace mlir::rvsdg;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RVSDG, rvsdg, RVSDGDialect)

//===---------------------------------------------------------------------===//
// ControlType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAControlType(MlirType type)
{
    return isa<ControlType>(unwrap(type));
}

MlirType mlirControlTypeGet(MlirContext ctx, uint64_t numOptions)
{
    return wrap(ControlType::get(unwrap(ctx), numOptions));
}
