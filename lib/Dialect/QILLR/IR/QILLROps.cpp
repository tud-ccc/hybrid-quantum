/// Implements the QILLR dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define DEBUG_TYPE "qillr-ops"

using namespace mlir;
using namespace mlir::qillr;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.cpp.inc"

//===----------------------------------------------------------------------===//
// QILLRDialect
//===----------------------------------------------------------------------===//

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

// QILLR gates do not return values
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

void QILLRDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.cpp.inc"
        >();
}
