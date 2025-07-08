/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUTypes.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Record.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

#define DEBUG_TYPE "qpu-ops"

using namespace mlir;
using namespace mlir::qpu;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//

// void DeviceOp::build(
//     OpBuilder &builder,
//     OperationState &state,
//     CouplingGraphAttr graphAttr)
// {
//     auto qubits = graphAttr.getQubits().getInt();
//     auto edges = graphAttr.getEdges();

//     // Construct result type using only raw values
//     auto type = qpu::DeviceType::get(builder.getContext(), qubits, edges);

//     build(builder, state, type, graphAttr);
// }

// LogicalResult DeviceOp::verify()
// {
//     auto device = getDevice().getType();
//     auto graph = getCouplingGraph();

//     if (graph.getQubits().getInt() != device.getQubits())
//         return emitOpError(
//             "Coupling graph's qubits and device's qubits do not "
//             "match");

//     if (graph.getEdges() != device.getEdges())
//         return emitOpError(
//             "Coupling graph's edges and device's edges do not "
//             "match");

//     return success();
// }

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//
CircuitOp CircuitOp::create(
    Location location,
    StringRef name,
    DeviceType device,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs)
{
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    CircuitOp::build(builder, state, name, device, type, attrs);
    return cast<CircuitOp>(Operation::create(state));
}

void CircuitOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    DeviceType device,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs,
    ArrayRef<DictionaryAttr> argAttrs)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        getDeviceTypeAttrName(state.name),
        TypeAttr::get(device));
    state.addAttribute(getCircuitTypeAttrName(state.name), TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty()) return;
    assert(type.getNumInputs() == argAttrs.size());
    // call_interface_impl
    function_interface_impl::addArgAndResultAttrs(
        builder,
        state,
        argAttrs,
        /*resultAttrs=*/std::nullopt,
        getArgAttrsAttrName(state.name),
        getResAttrsAttrName(state.name));
}
//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPUDialect
//===----------------------------------------------------------------------===//

void QPUDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.cpp.inc"
        >();
}
