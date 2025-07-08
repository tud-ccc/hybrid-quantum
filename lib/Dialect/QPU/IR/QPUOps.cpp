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

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

static ParseResult parseExecuteOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandNames,
    SmallVectorImpl<Type> &operandTypes)
{
    if (parser.parseOptionalKeyword("args")) return success();
    auto parseElement = [&]() -> ParseResult {
        return failure(
            parser.parseOperand(operandNames.emplace_back())
            || parser.parseColonType(operandTypes.emplace_back()));
    };

    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        parseElement,
        "in argument list");
}

static void printExecuteOperands(
    OpAsmPrinter &printer,
    Operation*,
    OperandRange operands,
    TypeRange types)
{
    if (operands.empty()) return;
    printer << "args(";
    llvm::interleaveComma(
        llvm::zip_equal(operands, types),
        printer,
        [&](const auto &pair) {
            auto [operand, type] = pair;
            printer << operand << " : " << type;
        });
    printer << ")";
}

static ParseResult parseExecuteResults(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandNames,
    SmallVectorImpl<Type> &operandTypes)
{
    if (parser.parseOptionalKeyword("outs")) return success();
    auto parseElement = [&]() -> ParseResult {
        return failure(
            parser.parseOperand(operandNames.emplace_back())
            || parser.parseColonType(operandTypes.emplace_back()));
    };

    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        parseElement,
        "in outs list");
}

static void printExecuteResults(
    OpAsmPrinter &printer,
    Operation*,
    OperandRange operands,
    TypeRange types)
{
    if (operands.empty()) return;
    printer << "outs(";
    llvm::interleaveComma(
        llvm::zip_equal(operands, types),
        printer,
        [&](const auto &pair) {
            auto [operand, type] = pair;
            printer << operand << " : " << type;
        });
    printer << ")";
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//
CircuitOp CircuitOp::create(
    Location location,
    StringRef name,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs)
{
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    CircuitOp::build(builder, state, name, type, attrs);
    return cast<CircuitOp>(Operation::create(state));
}

void CircuitOp::build(
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
// QPUDialect
//===----------------------------------------------------------------------===//

void QPUDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.cpp.inc"
        >();
}
