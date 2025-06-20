/// Implements the Quantum dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/ExtensibleDialect.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-types"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QubitType
//===----------------------------------------------------------------------===//

bool QubitType::isSingleQubit() const { return getSize() == 1; }

LogicalResult QubitType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    int64_t size)
{
    if (size < 1)
        return emitError() << "expected integer value greater equals 1";

    return success();
}

//===----------------------------------------------------------------------===//
// DeviceType
//===----------------------------------------------------------------------===//

bool DeviceType::hasStaticSize() const
{
    return getQubits() == ShapedType::kDynamic;
}

Type DeviceType::parse(::mlir::AsmParser &parser)
{
    if (parser.parseLess()) {
        parser.emitError(parser.getNameLoc(), "DeviceType starts with `<`");
        return Type();
    }

    int64_t qubits;
    if (!parser.parseOptionalInteger(qubits).has_value()) {
        if (!parser.parseQuestion()) {
            qubits = ShapedType::kDynamic;
        } else {
            parser.emitError(
                parser.getNameLoc(),
                "Device requires a qubit size.");
            return Type();
        }
    }

    if (parser.parseComma()) {
        parser.emitError(parser.getNameLoc(), "Device requires a topology.");
        return Type();
    }

    ArrayAttr edges;
    if (!parser.parseOptionalAttribute(edges, IntegerType()).has_value()) {
        if (!parser.parseQuestion()) {
            edges = ArrayAttr::get(
                parser.getContext(),
                ArrayRef<Attribute>(IntegerAttr::get(
                    IntegerType::get(parser.getContext(), 64),
                    ShapedType::kDynamic)));
        } else {
            parser.emitError(
                parser.getNameLoc(),
                "Device requires a topology.");
            return Type();
        }
    }

    if (parser.parseGreater()) {
        parser.emitError(parser.getNameLoc(), "DeviceType ends with `>`");
        return Type();
    }

    return DeviceType::get(parser.getContext(), qubits, edges);
}

void DeviceType::print(::mlir::AsmPrinter &printer) const {}

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"
        >();
}
