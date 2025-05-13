//===- DialectQIR.cpp - Pybind module for QIR dialect API support --===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/QIR.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectQIRSubmodule(nb::module_ m)
{
    auto qubitType = mlir_type_subclass(m, "QubitType", mlirTypeIsAQubitType);
}

NB_MODULE(_mlirDialectsQIR, m)
{
    m.doc() = "QIR dialect.";

    populateDialectQIRSubmodule(m);
}
