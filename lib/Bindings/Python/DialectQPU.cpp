//===---- DialectQPU.cpp - Pybind module for QPU dialect API support ------===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "quantum-mlir-c/Dialect/QPU.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <string>
#include <vector>

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
    auto targetAttr =
        mlir_attribute_subclass(m, "TargetAttr", mlirAttrIsATargetAttr);

    targetAttr.def_classmethod(
        "get",
        [](nb::object cls,
           MlirContext context,
           int64_t qubits,
           nb::ndarray<int64_t, nb::shape<-1, 2>> arr) {
            CollectDiagnosticsToStringScope scope(context);
            size_t rows = arr.shape(0);
            size_t cols = arr.shape(1);
            assert(cols == 2 && "Expected coupling to represent an edge list");
            // Build nested ArrayAttr: each row -> inner ArrayAttr of integer
            // attrs, then outer ArrayAttr contains all rows. This produces an
            // MlirAttribute (ArrayAttr) we can pass to the C API.
            const MlirType i64Type = mlirIntegerTypeGet(context, 64);

            std::vector<MlirAttribute> outerAttrs;
            outerAttrs.reserve(rows);

            for (size_t i = 0; i < rows; ++i) {
                std::vector<MlirAttribute> innerAttrs;
                innerAttrs.reserve(cols);
                for (size_t j = 0; j < cols; ++j) {
                    MlirAttribute intAttr =
                        mlirIntegerAttrGet(i64Type, arr(i, j));
                    innerAttrs.push_back(intAttr);
                }
                MlirAttribute innerArray = mlirArrayAttrGet(
                    context,
                    static_cast<intptr_t>(innerAttrs.size()),
                    innerAttrs.data());
                outerAttrs.push_back(innerArray);
            }

            MlirAttribute valuesAttr = mlirArrayAttrGet(
                context,
                static_cast<intptr_t>(outerAttrs.size()),
                outerAttrs.data());
            MlirAttribute qubitAttr = mlirIntegerAttrGet(i64Type, qubits);
            MlirAttribute attr =
                mlirTargetAttrGet(context, qubitAttr, valuesAttr);
            if (mlirAttributeIsNull(attr))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(attr);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("qubits"),
        nb::arg("values"));
}

NB_MODULE(_mlirDialectsQPU, m)
{
    m.doc() = "QPU dialect.";

    populateDialectQPUSubmodule(m);
}
