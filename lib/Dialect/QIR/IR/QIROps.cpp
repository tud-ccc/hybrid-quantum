/// Implements the QIR dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define DEBUG_TYPE "qir-ops"

using namespace mlir;
using namespace mlir::qir;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QIR/IR/QIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// QIRDialect
//===----------------------------------------------------------------------===//

// NOTE: We assume N qubit device that may or may not have passive qubits. The
// required qubits is thus the max qubit index regardless of topology.
LogicalResult AllocateDeviceOp::verify()
{
    int64_t num_qubits = getNumQubits();
    ArrayAttr coupling_graph = getCouplingGraph();

    // Check if the coupling graph is valid
    for (Attribute edge : coupling_graph) {
        // Each edge should be an array of two integers
        ArrayAttr edgeArray = dyn_cast<ArrayAttr>(edge);
        if (!edgeArray || edgeArray.size() != 2)
            return emitOpError(
                "each edge in coupling graph must be an array of two integers");

        for (Attribute qubit : edgeArray) {
            IntegerAttr qubitAttr = dyn_cast<IntegerAttr>(qubit);
            if (!qubitAttr)
                return emitOpError("each qubit in edge must be an integer");
            int64_t q = qubitAttr.getInt();
            if (q < 0 || q >= num_qubits)
                return emitOpError(
                    "qubit index " + Twine(q)
                    + " is out of bounds for device with " + Twine(num_qubits)
                    + " qubits");
        }
    }
    return success();
}

//=----------------------------------------------------------------------===//
// GateCallOp
//===----------------------------------------------------------------------===//

LogicalResult GateCallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    auto gateNameAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!gateNameAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");

    GateOp gate =
        symbolTable.lookupNearestSymbolFrom<GateOp>(*this, gateNameAttr);
    if (!gate)
        return emitOpError() << "'" << gateNameAttr.getValue()
                             << "' does not reference a valid function";

    return success();
}

// QIR gates do not return values
FunctionType GateCallOp::getCalleeType()
{
    return FunctionType::get(
        getContext(),
        getOperandTypes(),
        {}); // getResultTypes()
}

//===----------------------------------------------------------------------===//
// GateOp
//===----------------------------------------------------------------------===//

GateOp GateOp::create(
    Location location,
    StringRef name,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs)
{
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    GateOp::build(builder, state, name, type, attrs);
    return cast<GateOp>(Operation::create(state));
}

void GateOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs,
    ArrayRef<DictionaryAttr> argAttrs)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(type));
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

void QIRDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/QIR/IR/QIROps.cpp.inc"
        >();
}
