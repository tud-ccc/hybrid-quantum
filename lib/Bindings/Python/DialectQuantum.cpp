//===- DialectQuantum.cpp - Pybind module for Quantum dialect API support -===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/Quantum.h"

#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectQuantumSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // Quantum dialect registration
    //===--------------------------------------------------------------------===//
    auto quantum = m.def_submodule("quantum");

    quantum.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__quantum__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    //===--------------------------------------------------------------------===//
    // QubitType
    //===--------------------------------------------------------------------===//
    auto qubitType =
        mlir_type_subclass(m, "QuantumQubitType", mlirTypeIsAQuantumQubitType);

    qubitType.def_classmethod(
        "get",
        [](nb::object cls, MlirContext context, uint64_t size) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirQuantumQubitTypeGet(context, size);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("size"));
}

NB_MODULE(_mlirDialectsQuantum, m)
{
    m.doc() = "Quantum dialect.";

    populateDialectQuantumSubmodule(m);
}
