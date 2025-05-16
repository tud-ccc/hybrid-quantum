//===- DialectQIR.cpp - Pybind module for QIR dialect API support --===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/QIR.h"

#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectQIRSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // QIR dialect registration
    //===--------------------------------------------------------------------===//
    auto qir = m.def_submodule("qir");

    qir.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__qir__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    //===--------------------------------------------------------------------===//
    // Qubit Type
    //===--------------------------------------------------------------------===//
    auto qubitType = mlir_type_subclass(m, "QubitType", mlirTypeIsAQubitType);

    qubitType.def_classmethod(
        "get",
        [](nb::object cls, MlirContext context) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirQubitTypeGet(context);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none());
}

NB_MODULE(_mlirDialectsQIR, m)
{
    m.doc() = "QIR dialect.";

    populateDialectQIRSubmodule(m);
}
