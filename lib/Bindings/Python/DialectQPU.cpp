//===---- DialectQPU.cpp - Pybind module for QPU dialect API support ------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/QPU.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectQPUSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // QPU dialect registration
    //===--------------------------------------------------------------------===//
    auto dialect = m.def_submodule("qpu");

    dialect.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__qpu__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    //===--------------------------------------------------------------------===//
    // TargetAttr
    //===--------------------------------------------------------------------===//
    auto matchRule =
        mlir_attribute_subclass(m, "MatchRuleAttr", mlirAttrIsAMatchRuleAttr);

    matchRule.def_classmethod(
        "get",
        [](nb::object cls,
           MlirContext context,
           nb::ndarray<int, nb::shape<-1, 2>> arr,
           uint64_t index) {
            CollectDiagnosticsToStringScope scope(context);
            size_t rows = arr.shape(0);
            size_t cols = arr.shape(1);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    int value = arr(i, j);
                    // do something
                }
            }

            MlirAttribute attr = mlirTargetAttrGet(context, arr, index);
            if (mlirAttributeIsNull(attr))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(attr);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("values"),
        nb::arg("index"));
}

NB_MODULE(_mlirDialectsRVSDG, m)
{
    m.doc() = "RVSDG dialect.";

    populateDialectRVSDGSubmodule(m);
}
