//===- DialectRVSDG.cpp - Pybind module for RVSDG dialect API support -----===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/RVSDG.h"

#include <cstdint>
#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectRVSDGSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // RVSDG dialect registration
    //===--------------------------------------------------------------------===//
    auto rvsdg = m.def_submodule("rvsdg");

    rvsdg.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__rvsdg__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    //===--------------------------------------------------------------------===//
    // ControlType
    //===--------------------------------------------------------------------===//
    auto controlType =
        mlir_type_subclass(m, "ControlType", mlirTypeIsAControlType);

    controlType.def_classmethod(
        "get",
        [](nb::object cls, MlirContext context, uint64_t numOptions) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirControlTypeGet(context, numOptions);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("numOptions"));
}

NB_MODULE(_mlirDialectsRVSDG, m)
{
    m.doc() = "RVSDG dialect.";

    populateDialectRVSDGSubmodule(m);
}
