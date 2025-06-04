/// Implements the Quantum dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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
        auto theta1 = op.getTheta();
        auto theta2 = rz.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        rewriter.eraseOp(op);
        rewriter.replaceOpWithNewOp<RzOp>(
            rz,
            rz.getInput(),
            thetaPlus.getResult());

        return success();
    }
    return failure();
}

LogicalResult RxOp::canonicalize(RxOp op, PatternRewriter &rewriter)
{
    // %1 = Rx(%0, %theta1)
    // %2 = Rx(%1, %theta2)
    // --------------------
    // %1 = Rx(%0, %theta1 + %theta2)
    if (auto rx = op.getInput().getDefiningOp<RxOp>()) {
        auto theta1 = op.getTheta();
        auto theta2 = rx.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        rewriter.eraseOp(op);
        rewriter.replaceOpWithNewOp<RxOp>(
            rx,
            rx.getInput(),
            thetaPlus.getResult());

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
        auto theta1 = op.getTheta();
        auto theta2 = ry.getTheta();

        auto loc = op.getLoc();
        auto thetaPlus = rewriter.create<arith::AddFOp>(loc, theta1, theta2);

        rewriter.eraseOp(op);
        rewriter.replaceOpWithNewOp<RyOp>(
            ry,
            ry.getInput(),
            thetaPlus.getResult());

        return success();
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

template<typename ConcreteType>
LogicalResult NoClone<ConcreteType>::verifyTrait(Operation* op)
{
    // Check whether the qubits capured by the IfOp
    // are used more than a single time in each region
    if (auto ifOp = llvm::dyn_cast_if_present<quantum::IfOp>(op)) {
        // Check `thenRegion`
        Region &thenRegion = ifOp.getThenRegion();
        Block &thenBlock = thenRegion.getBlocks().front();
        for (auto value : thenBlock.getArguments()) {
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

        // Check optional `elseRegion`
        Region &elseRegion = ifOp.getElseRegion();
        if (!elseRegion.empty()) {
            Block &elseBlock = elseRegion.getBlocks().front();
            for (auto value : elseBlock.getArguments()) {
                auto uses = value.getUses();
                int numUses = std::distance(uses.begin(), uses.end());
                if (numUses > 1) {
                    return op->emitOpError()
                           << "captured qubit #" << value.getArgNumber()
                           << " used more than once within the same block";
                }
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

LogicalResult IfOp::verify()
{
    if (getNumResults() != 0 && getElseRegion().empty())
        return emitOpError("must have an else block if defining values");

    if (getNumRegionCapturedArgs() != getNumResults())
        return emitOpError("# return values != # captured values");

    return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer'
/// values are regular SSA values.
static void printInitializationList(
    OpAsmPrinter &p,
    Block::BlockArgListType blocksArgs,
    ValueRange initializers,
    StringRef prefix = "")
{
    // the block arguments will be the conditional (1) + the list of
    // initializers
    assert(
        blocksArgs.size() == initializers.size()
        && "expected same length of arguments and initializers");
    if (initializers.empty()) return;

    p << prefix << '(';
    llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
        p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ")";
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir::quantum::buildTerminatedBody(
    OpBuilder &builder,
    Location loc,
    Value condition,
    ValueRange capturedArgs)
{
    builder.create<quantum::YieldOp>(loc);
}

Block* IfOp::thenBlock()
{
    Block* thenBlock = &getThenRegion().getBlocks().front();
    return thenBlock;
}

Block* IfOp::elseBlock()
{
    if (getElseRegion().empty()) return nullptr;
    Block* elseBlock = &getElseRegion().getBlocks().front();
    return elseBlock;
}

void IfOp::build(
    OpBuilder &builder,
    OperationState &result,
    Value condition,
    ValueRange capturedArgs,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> thenBuilder,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> elseBuilder)
{
    // Build then region.
    // The thenBuilder is required.
    assert(thenBuilder && "the builder callback for 'then' must be present");
    OpBuilder::InsertionGuard guard(builder);
    result.addOperands(condition);
    if (!capturedArgs.empty()) result.addOperands(capturedArgs);

    for (Value v : capturedArgs) result.addTypes(v.getType());

    result.regions.reserve(2);
    Region* thenRegion = result.addRegion();
    Block* thenBlock = builder.createBlock(thenRegion);
    // thenBlock->addArgument(condition.getType(), result.location);
    for (Value v : capturedArgs)
        thenBlock->addArguments(v.getType(), v.getLoc());

    // Always true.
    if (thenBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(thenBlock);
        thenBuilder(
            builder,
            result.location,
            condition,
            thenBlock->getArguments());
    }

    if (capturedArgs.empty())
        IfOp::ensureTerminator(*thenRegion, builder, result.location);

    // Build the else region.
    // The elseBuilder is optional.
    Region* elseRegion = result.addRegion();
    if (elseBuilder) {
        Block* elseBlock = builder.createBlock(elseRegion);
        // elseBlock->addArgument(condition.getType(), result.location);
        for (Value v : capturedArgs)
            elseBlock->addArguments(v.getType(), v.getLoc());

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(elseBlock);
        elseBuilder(
            builder,
            result.location,
            condition,
            elseBlock->getArguments());

        if (capturedArgs.empty())
            IfOp::ensureTerminator(*elseRegion, builder, result.location);
    }
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result)
{

    // Reserve the regions for `then` and `else`
    result.regions.reserve(2);
    Region* thenRegion = result.addRegion();
    Region* elseRegion = result.addRegion();

    auto &builder = parser.getBuilder();

    // Parse the condition
    OpAsmParser::UnresolvedOperand conditionVariable;
    Type i1Type = builder.getIntegerType(1);
    if (parser.parseOperand(conditionVariable)
        || parser.resolveOperand(conditionVariable, i1Type, result.operands))
        return failure();

    // Parse the capture list of qubits
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

    bool isCapturingArgs = succeeded(parser.parseOptionalKeyword("ins"));
    if (isCapturingArgs) {
        // Parse the assignment list
        if (parser.parseAssignmentList(regionArgs, operands)) return failure();
    }

    // Parse the result type list
    if (parser.parseArrowTypeList(result.types)) return failure();

    // Set the block argument types for the captured operands
    for (auto [capturedArg, type] : llvm::zip_equal(regionArgs, result.types))
        capturedArg.type = type;

    if (regionArgs.size() != result.types.size())
        return parser.emitError(
            parser.getNameLoc(),
            "mismatch in number of captured values and defined values");

    // Parse the `then` region
    if (parser.parseRegion(*thenRegion, regionArgs)) return failure();
    quantum::IfOp::ensureTerminator(*thenRegion, builder, result.location);

    // Parse the optional `else` region
    if (!parser.parseOptionalKeyword("else")) {
        if (parser.parseRegion(*elseRegion, regionArgs)) return failure();
        quantum::IfOp::ensureTerminator(*elseRegion, builder, result.location);
    }

    if (isCapturingArgs) {
        for (auto argOperandType :
             llvm::zip_equal(regionArgs, operands, result.types)) {
            Type type = std::get<2>(argOperandType);
            std::get<0>(argOperandType).type = type;
            if (parser.resolveOperand(
                    std::get<1>(argOperandType),
                    type,
                    result.operands))
                return failure();
        }
    }

    // Parse the optional attribute list.
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();

    return success();
}

void IfOp::print(OpAsmPrinter &p)
{
    p << " " << getCondition();

    printInitializationList(
        p,
        getRegionCapturedArgs(),
        getCapturedArgs(),
        " ins");

    if (!getResults().empty()) p << " -> (" << getResultTypes() << ")";
    p << ' ';
    p.printRegion(
        getThenRegion(),
        /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/true);

    // Print the 'else' regions if it exists and has a block.
    auto &elseRegion = getElseRegion();
    if (!elseRegion.empty()) {
        p << " else ";
        p.printRegion(
            elseRegion,
            /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true);
    }

    p.printOptionalAttrDict((*this)->getAttrs());
}

void IfOp::getSuccessorRegions(
    RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions)
{
    // The `then` and the `else` region branch back to the parent operation.
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor(getResults()));
        return;
    }

    regions.push_back(RegionSuccessor(&getThenRegion()));

    // Don't consider the else region if it is empty.
    Region* elseRegion = &this->getElseRegion();
    if (elseRegion->empty())
        regions.push_back(RegionSuccessor());
    else
        regions.push_back(RegionSuccessor(elseRegion));
}

void IfOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions)
{
    FoldAdaptor adaptor(operands, *this);
    auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
    if (!boolAttr || boolAttr.getValue())
        regions.emplace_back(&getThenRegion());

    // If the else region is empty, execution continues after the parent op.
    if (!boolAttr || !boolAttr.getValue()) {
        if (!getElseRegion().empty())
            regions.emplace_back(&getElseRegion());
        else
            regions.emplace_back(getResults());
    }
}

void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds)
{
    if (auto cond = llvm::dyn_cast_or_null<BoolAttr>(operands[0])) {
        // If the condition is known, then one region is known to be
        // executed once and the other zero times.
        invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
        invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
    } else {
        // Non-constant condition. Each region may be executed 0 or 1 times.
        invocationBounds.assign(2, {0, 1});
    }
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
