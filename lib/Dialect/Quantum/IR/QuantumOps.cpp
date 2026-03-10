/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumAttributes.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Record.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
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
// QPUDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with gate operations.
struct QuantumInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// Call operations can always be inlined
    bool isLegalToInline(Operation*, Operation*, bool) const final
    {
        return true;
    }

    /// All operations can be inlined.
    bool isLegalToInline(Operation*, Region*, bool, IRMapping &) const final
    {
        return true;
    }

    /// All gate bodies can be inlined.
    bool isLegalToInline(Region*, Region*, bool, IRMapping &) const final
    {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator by replacing it with a new
    /// operation as necessary.
    void handleTerminator(Operation* op, Block*) const final
    {
        auto returnOp = llvm::dyn_cast<quantum::ReturnOp>(op);
        if (!returnOp) return;

        returnOp->erase();
    };

    /// Handle the given inlined terminator by replacing its operands
    void handleTerminator(Operation* op, ValueRange valuesToRepl) const final
    {
        // Only "quantum.return" needs to be handled here.
        auto returnOp = llvm::dyn_cast<quantum::ReturnOp>(op);
        if (!returnOp) return;

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    };
};
} // namespace

//===--------------------------------------------------------------------===//
// InterRegisterRangesInterface Hooks
//===--------------------------------------------------------------------===//

void AllocOp::inferResultRanges(
    ArrayRef<RegisterRanges>,
    SetRangeFn setResultRanges)
{
    auto zero = llvm::APInt(64, 0);
    auto size = llvm::APInt(64, getResult().getType().getSize());
    ConstantIntRanges range(zero, size, zero, size);
    setResultRanges(
        getResult(),
        RegisterRanges(ConstantRegisterRanges(getResult(), range)));
}

void DeallocateOp::inferResultRanges(ArrayRef<RegisterRanges>, SetRangeFn)
{
    return;
}

void SplitOp::inferResultRanges(
    ArrayRef<RegisterRanges> argRanges,
    SetRangeFn setResultRanges)
{
    // Split op takes an input and returns multiple outputs each holding a
    // subintervals
    assert(argRanges.size() == 1 && "Only expect a single input to split");
    RegisterRanges splitRange = argRanges[0];
    for (auto result : getResults()) {
        auto ty = llvm::dyn_cast_if_present<QubitType>(result.getType());
        if (!ty) continue;
        auto resRange = splitRange.take_front(ty.getSize());
        splitRange = splitRange.drop_front(ty.getSize());
        setResultRanges(result, resRange);
    }
}

void MergeOp::inferResultRanges(
    ArrayRef<RegisterRanges> argRanges,
    SetRangeFn setResultRanges)
{
    RegisterRanges result;
    for (RegisterRanges arg : argRanges)
        result = RegisterRanges::join(result, arg);

    setResultRanges(getResult(), result);
}

void MeasureOp::inferResultRanges(
    ArrayRef<RegisterRanges> argRanges,
    SetRangeFn setResultRanges)
{
    setResultRanges(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

template<typename OpTy>
LogicalResult
controlledRotationOpCanonicalize(OpTy op, PatternRewriter &rewriter)
{
    // %ctrl1, %1 = CR_(%ctrl, %0, %theta1)
    // %ctrl2, %2 = CR_(%ctrl1, %1, %theta2)
    // --------------------
    // %ctrl1, %1 = R_(%ctrl, %0, %theta1 + %theta2)

    // Ctrl + Input from same op
    if (op.getTarget().getDefiningOp() == op.getControl().getDefiningOp()) {
        if (auto otherRotation =
                op.getTarget().template getDefiningOp<OpTy>()) {
            // addf either folds the constant folded values or the result of
            // addf
            llvm::SmallVector<Value, 2> addfv;
            rewriter.createOrFold<arith::AddFOp>(
                addfv,
                op->getLoc(),
                otherRotation.getAngle(),
                op.getAngle());

            auto addf = addfv.front();
            if (auto fconst = llvm::dyn_cast<arith::ConstantFloatOp>(
                    addf.getDefiningOp())) {
                auto fattr = llvm::cast<FloatAttr>(fconst.getValueAttr());
                auto apfloat = fattr.getValue();
                // If op.theta + other.theta = -0.0 or +0.0
                // Then we can remove both rotations
                if (apfloat.isZero()) {
                    rewriter.replaceAllUsesWith(
                        op.getResults(),
                        {otherRotation.getControl(),
                         otherRotation.getTarget()});
                    rewriter.eraseOp(op);
                    return success();
                }
            }

            rewriter.replaceOpWithNewOp<OpTy>(
                op,
                otherRotation.getControl(),
                otherRotation.getTarget(),
                addf);
            rewriter.eraseOp(otherRotation);
            return success();
        }
    }
    return failure();
}

template<typename OpTy>
LogicalResult rotationOpCanonicalize(OpTy op, PatternRewriter &rewriter)
{
    // %1 = R_(%0, %theta1)
    // %2 = R_(%1, %theta2)
    // --------------------
    // %1 = R_(%0, %theta1 + %theta2)
    if (auto otherRotation = op.getInput().template getDefiningOp<OpTy>()) {
        // addf either folds the constant folded values or the result of addf
        llvm::SmallVector<Value, 2> addfv;
        rewriter.createOrFold<arith::AddFOp>(
            addfv,
            op->getLoc(),
            otherRotation.getTheta(),
            op.getTheta());

        auto addf = addfv.front();
        if (auto fconst =
                llvm::dyn_cast<arith::ConstantFloatOp>(addf.getDefiningOp())) {
            auto fattr = llvm::cast<FloatAttr>(fconst.getValueAttr());
            auto apfloat = fattr.getValue();
            // If op.theta + other.theta = -0.0 or +0.0
            // Then we can remove both rotations
            if (apfloat.isZero()) {
                rewriter.replaceAllUsesWith(
                    op.getResult(),
                    otherRotation.getInput());
                rewriter.eraseOp(op);
                return success();
            }
        }

        rewriter.replaceOpWithNewOp<OpTy>(op, otherRotation.getInput(), addf);
        rewriter.eraseOp(otherRotation);
        return success();
    }
    return failure();
}
} // namespace

LogicalResult CRzOp::canonicalize(CRzOp op, PatternRewriter &rewriter)
{
    return controlledRotationOpCanonicalize(op, rewriter);
}

LogicalResult CRyOp::canonicalize(CRyOp op, PatternRewriter &rewriter)
{
    return controlledRotationOpCanonicalize(op, rewriter);
}

LogicalResult RzOp::canonicalize(RzOp op, PatternRewriter &rewriter)
{
    return rotationOpCanonicalize(op, rewriter);
}

LogicalResult RxOp::canonicalize(RxOp op, PatternRewriter &rewriter)
{
    return rotationOpCanonicalize(op, rewriter);
}

LogicalResult RyOp::canonicalize(RyOp op, PatternRewriter &rewriter)
{
    return rotationOpCanonicalize(op, rewriter);
}

LogicalResult SXOp::canonicalize(SXOp op, PatternRewriter &rewriter)
{
    // q1 = SX(q); q2 = SX(q1) =: q2 = X(q)
    if (auto otherOp = llvm::dyn_cast<SXOp>(op.getInput().getDefiningOp())) {
        rewriter.replaceOpWithNewOp<XOp>(op, otherOp.getInput());
        rewriter.eraseOp(otherOp);
        return success();
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// Folders
//===----------------------------------------------------------------------===//

// Free function implementation of foldTrait
template<typename ConcreteType>
LogicalResult foldHermitianTraitImpl(
    Operation* op,
    ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult> &results)
{
    if (op->getNumOperands() == 1) {
        // Check for H * H
        if (matchPattern(op->getOperand(0), m_Op<ConcreteType>())) {
            // Add the other's operand to the results vector
            auto otherOp = op->getOperand(0).getDefiningOp();
            results.push_back(otherOp->getOperand(0));
            return success();
        }
    } else if (
        // x, z = CNOT(a, b); CNOT(x, z); := a, b
        matchPattern(op->getOperand(0), m_Op<ConcreteType>())
        && matchPattern(op->getOperand(1), m_Op<ConcreteType>())) {
        if (op->getOperand(0).getDefiningOp()
            == op->getOperand(1).getDefiningOp()) {
            // Add the other's operands to the results vector
            auto otherOp = op->getOperand(0).getDefiningOp();
            results.push_back(otherOp->getOperand(0));
            results.push_back(otherOp->getOperand(1));
            return success();
        }
    }
    return failure();
}

/// Override the 'foldTrait' hook to support trait based folding on the
/// concrete operation.
template<typename ConcreteType>
LogicalResult Hermitian<ConcreteType>::foldTrait(
    Operation* op,
    ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult> &results)
{
    return foldHermitianTraitImpl<ConcreteType>(op, operands, results);
}

// Free function implementation of foldTrait
template<typename ConcreteType>
LogicalResult foldUnitaryTraitImpl(
    Operation* op,
    ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult> &results)
{
    if (op->getNumOperands() == 1) {
        // q1 = adjoint(op)(q); q2 = op(q1)
        // OR q1 = op; q2 = adjoint(op)(q1)
        auto otherOp = op->getOperand(0).getDefiningOp();
        if (otherOp->hasTrait<AdjointTo<ConcreteType>::template Impl>()) {
            results.push_back(otherOp->getOperand(0));
            return success();
        }
        return failure();
    }
    return failure();
}

/// Override the 'foldTrait' hook to support trait based folding on the
/// concrete operation.
template<typename ConcreteType>
LogicalResult Unitary<ConcreteType>::foldTrait(
    Operation* op,
    ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult> &results)
{
    return foldUnitaryTraitImpl<ConcreteType>(op, operands, results);
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

LogicalResult MergeOp::verify()
{
    int64_t size = 0;
    for (auto operand : getOperands()) {
        auto in = llvm::cast<QubitType>(operand.getType());
        size += in.getSize();
    }
    if (getResult().getType().getSize() != size)
        return emitOpError(
            "result size must be equal to sum of operand sizes.");

    return success();
}

LogicalResult SplitOp::verify()
{
    int64_t size = 0;
    for (auto result : getResults()) {
        auto out = llvm::cast<QubitType>(result.getType());
        size += out.getSize();
    }
    if (getInput().getType().getSize() != size)
        return emitOpError(
            "operand size must be equal to sum of result sizes.");

    return success();
}

// NOTE: We assume N qubit device that may or may not have passive qubits.
// The required qubits is thus the max qubit index regardless of topology.
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
//                 return emitOpError("each qubit in edge must be an
//                 integer");
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
    static auto sumUses = [](auto uses) {
        return std::distance(uses.begin(), uses.end());
    };

    for (auto &value : op->getOpOperands()) {
        // Ignore captured non-qubit types
        if (!llvm::dyn_cast<quantum::QubitType>(value.get().getType()))
            continue;
        auto numUses = sumUses(value.get().getUses());
        if (numUses > 1)
            return op->emitOpError() << "qubit #" << value.getOperandNumber()
                                     << " is used " << numUses << " times.";
    }

    // Check whether the qubit values returned from an operation
    // are uses more than a single time.
    for (auto value : op->getOpResults()) {
        if (!llvm::isa<quantum::QubitType>(value.getType())) continue;
        if (sumUses(value.getUses()) > 1) {
            return op->emitOpError()
                   << "result qubit #" << value.getResultNumber()
                   << " used more than once within the same block";
        }
    }

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
    addInterfaces<QuantumInlinerInterface>();
}
