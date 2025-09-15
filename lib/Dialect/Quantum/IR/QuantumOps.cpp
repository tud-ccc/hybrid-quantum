/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumAttributes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

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

#define DEBUG_TYPE "quantum-ops"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//
LogicalResult RzOp::canonicalize(RzOp op, PatternRewriter &rewriter)
{
    // %1 = Rz(%0, %theta1)
    // %2 = Rz(%1, %theta2)
    // --------------------
    // %1 = Rz(%0, %theta1 + %theta2)
    if (auto rz = op.getInput().getDefiningOp<RzOp>()) {
        auto theta1 = rz.getTheta();
        auto theta2 = op.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        auto newRz = rewriter.replaceOpWithNewOp<RzOp>(
            rz,
            rz.getInput(),
            thetaPlus.getResult());
        op->replaceAllUsesWith(newRz->getResults());
        rewriter.eraseOp(op);

        return success();
    }
    return failure();
}

LogicalResult RxOp::canonicalize(RxOp op, PatternRewriter &rewriter)
{
    // %1 = Rx(%0, %theta1) -> rx
    // %2 = Rx(%1, %theta2) -> op
    // --------------------
    // %1 = Rx(%0, %theta1 + %theta2)
    if (auto rx = op.getInput().getDefiningOp<RxOp>()) {
        auto theta1 = rx.getTheta();
        auto theta2 = op.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        auto newRx = rewriter.replaceOpWithNewOp<RxOp>(
            rx,
            rx.getInput(),
            thetaPlus.getResult());
        op->replaceAllUsesWith(newRx->getResults());
        rewriter.eraseOp(op);

        return success();
    }
    return failure();
}

LogicalResult RyOp::canonicalize(RyOp op, PatternRewriter &rewriter)
{
    // %1 = Ry(%0, %theta1)
    // %2 = Ry(%1, %theta2)
    // --------------------
    // %1 = Ry(%0, %theta1 + %theta2)
    if (auto ry = op.getInput().getDefiningOp<RyOp>()) {
        auto theta1 = ry.getTheta();
        auto theta2 = op.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        auto newRy = rewriter.replaceOpWithNewOp<RyOp>(
            ry,
            ry.getInput(),
            thetaPlus.getResult());
        op->replaceAllUsesWith(newRy->getResults());
        rewriter.eraseOp(op);

        return success();
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

// NOTE: We assume N qubit device that may or may not have passive qubits. The
// required qubits is thus the max qubit index regardless of topology.
// LogicalResult DeviceOp::verify()
// {
//     int64_t num_qubits = getQubits();
//     ArrayAttr coupling_graph = getCouplingGraph();

//     // Check if the coupling graph is valid
//     for (Attribute edge : coupling_graph) {
//         // Each edge should be an array of two integers
//         ArrayAttr edgeArray = dyn_cast<ArrayAttr>(edge);
//         if (!edgeArray || edgeArray.size() != 2)
//             return emitOpError(
//                 "each edge in coupling graph must be an array of two
//                 integers");

//         for (Attribute qubit : edgeArray) {
//             IntegerAttr qubitAttr = dyn_cast<IntegerAttr>(qubit);
//             if (!qubitAttr)
//                 return emitOpError("each qubit in edge must be an integer");
//             int64_t q = qubitAttr.getInt();
//             if (q < 0 || q >= num_qubits)
//                 return emitOpError(
//                     "qubit index " + Twine(q)
//                     + " is out of bounds for device with " +
//                     Twine(num_qubits)
//                     + " qubits");
//         }
//     }
//     return success();
// }

template<typename ConcreteType>
LogicalResult NoClone<ConcreteType>::verifyTrait(Operation* op)
{
    // For a region check if the region args are used more than once
    for (auto &region : op->getRegions()) {
        Block &block = region.getBlocks().front();
        for (auto value : block.getArguments()) {
            // Ignore captured non-qubit types
            if (!llvm::dyn_cast<quantum::QubitType>(value.getType())) continue;
            auto uses = value.getUses();
            int numUses = std::distance(uses.begin(), uses.end());
            if (numUses > 1) {
                return op->emitOpError()
                       << "captured qubit #" << value.getArgNumber()
                       << " used more than once within the same block";
            }
        }
    }

    // Check whether the qubit values returned from an operation
    // are uses more than a single time.
    for (auto value : op->getOpResults()) {
        if (!llvm::isa<quantum::QubitType>(value.getType())) continue;
        auto uses = value.getUses();
        int numUses = std::distance(uses.begin(), uses.end());
        if (numUses > 1) {
            return op->emitOpError()
                   << "result qubit #" << value.getResultNumber()
                   << " used more than once within the same block";
        }
    }

    return success();
}

template<typename ConcreteType>
LogicalResult Hermitian<ConcreteType>::verifyTrait(Operation* op)
{
    if (op->getNumOperands() != op->getNumResults())
        return op->emitOpError(
            "must have the same number of operands and results");

    return success();
}

LogicalResult ReturnOp::verify()
{
    auto returnedValuesEqualSize =
        [&](OperationName name, ArrayRef<Type> results) -> LogicalResult {
        if (getNumOperands() != results.size())
            return emitOpError("has ")
                   << getNumOperands() << " operands, but enclosing function (@"
                   << name << ") returns " << results.size();

        for (unsigned i = 0, e = results.size(); i != e; ++i)
            if (getOperand(i).getType() != results[i])
                return emitError() << "type of return operand " << i << " ("
                                   << getOperand(i).getType()
                                   << ") doesn't match function result type ("
                                   << results[i] << ")"
                                   << " in function @" << name;

        return success();
    };

    if (auto customGate = dyn_cast<GateOp>((*this)->getParentOp())) {
        auto check = returnedValuesEqualSize(
            customGate->getName(),
            customGate.getFunctionType().getResults());
        if (failed(check)) return check;
    }

    return success();
}

//===----------------------------------------------------------------------===//
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

FunctionType GateCallOp::getCalleeType()
{
    return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
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

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
        >();
}
