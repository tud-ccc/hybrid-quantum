/// Implements the RVSDG dialect ops.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

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

#define DEBUG_TYPE "rvsdg-ops"

using namespace mlir;
using namespace mlir::rvsdg;

//===----------------------------------------------------------------------===//
// Custom parse and print methods
//===----------------------------------------------------------------------===//

static ParseResult parseTypedParamList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandNames,
    SmallVectorImpl<Type> &operandTypes)
{
    auto parseElement = [&]() -> ParseResult {
        return failure(
            parser.parseOperand(operandNames.emplace_back())
            || parser.parseColonType(operandTypes.emplace_back()));
    };

    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        parseElement,
        "in argument list");

    return ParseResult::success();
}

static ParseResult parseRVSDGRegion(OpAsmParser &parser, Region &region)
{
    SmallVector<OpAsmParser::Argument, 4> arguments;
    if (failed(parser.parseArgumentList(
            arguments,
            OpAsmParser::Delimiter::Paren,
            true,
            true))) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "Failed to parse argument list");
    }
    if (failed(parser.parseColon())) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "Expected a \":\" token");
    }

    if (failed(parser.parseRegion(region, arguments, true))) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "Failed to parse region");
    }
    return ParseResult::success();
}

static ParseResult parseRVSDGRegions(
    OpAsmParser &parser,
    SmallVectorImpl<std::unique_ptr<Region>> &regions)
{
    auto parseRegion = [&]() -> ParseResult {
        std::unique_ptr<Region> region = std::make_unique<Region>();
        if (failed(parseRVSDGRegion(parser, *region))) return failure();
        regions.push_back(std::move(region));
        return success();
    };

    if (parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::Square,
            parseRegion)) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "Failed to parse region list");
    }
    return success();
}

static void printRVSDGRegion(OpAsmPrinter &p, Operation* op, Region &region)
{
    p << "(";
    size_t numArguments = region.getNumArguments();
    for (size_t index = 0; index < numArguments; ++index) {
        if (index != 0) p << ", ";
        p.printRegionArgument(region.getArgument(index));
    }
    p << "): ";
    p.printRegion(region, false, true, true);
}

static void printRVSDGRegions(
    OpAsmPrinter &p,
    Operation* op,
    MutableArrayRef<Region> regions)
{
    p.increaseIndent();
    p << "[";
    p.printNewline();
    size_t numRegions = regions.size();
    for (size_t index = 0; index < numRegions; ++index) {
        if (index != 0) {
            p << ", ";
            p.printNewline();
        }
        printRVSDGRegion(p, op, regions[index]);
    }
    p.decreaseIndent();
    p.printNewline();
    p << "]";
}

static void printTypedParamList(
    OpAsmPrinter &p,
    Operation* op,
    OperandRange operands,
    TypeRange types)
{
    p << "(";
    int numParams = std::min(operands.size(), types.size());
    for (int i = 0; i < numParams; ++i) {
        if (i != 0) p << ", ";
        p.printOperand(operands[i]);
        p << ": ";
        p.printType(types[i]);
    }
    p << ")";
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

LogicalResult GammaNode::verify()
{
    if (this->getNumRegions() < 2) {
        return emitOpError(
                   "has too few regions. Minimum number of regions is 2, "
                   "but Op has ")
               << this->getNumRegions();
    }
    auto predicateType = this->getPredicate().getType();
    if (predicateType.getNumOptions() != this->getNumRegions()) {
        return emitOpError(
                   "has predicate with wrong number of options. Expected ")
               << this->getNumRegions() << ", got "
               << predicateType.getNumOptions();
    }
    for (auto &region : this->getRegions()) {
        if (region.getNumArguments() != this->getInputs().size()) {
            return emitOpError(
                       " has region with wrong number of arguments. "
                       "Offending region: #")
                   << region.getRegionNumber() << ". Expected "
                   << this->getInputs().size() << ", got "
                   << region.getNumArguments();
        }
        auto arguments = region.getArguments();
        auto inputs = this->getInputs();
        for (size_t i = 0; i < region.getNumArguments(); ++i) {
            if (arguments[i].getType() != inputs[i].getType()) {
                auto argument = arguments[i];
                emitOpError(" has mismatched region argument types: Region #")
                    << region.getRegionNumber() << " Argument #"
                    << argument.getArgNumber() << ". Expected "
                    << inputs[i].getType() << ", got "
                    << arguments[i].getType();
            }
        }
    }
    return LogicalResult::success();
}

LogicalResult YieldOp::verify()
{
    auto parent = llvm::dyn_cast<GammaNode>((*this)->getParentOp());
    if (parent == NULL)
        return emitOpError("YieldOp has no parent of type GammaNode.");

    const auto &results = parent.getResults();
    if (getNumOperands() != results.size()) {
        return emitOpError("has ")
               << getNumOperands() << " operands, but parent node outputs "
               << results.size();
    }

    for (size_t i = 0; i < results.size(); ++i) {
        if (getOperand(i).getType() != results[i].getType()) {
            return emitError() << "type of output operand " << i << " ("
                               << getOperand(i).getType()
                               << ") does not match node output type ("
                               << results[i].getType() << ")";
        }
    }

    return success();
}

LogicalResult Match::verify()
{
    auto mappingAttr = this->getMapping();
    auto numOptions = this->getOutput().getType().getNumOptions();

    std::unordered_map<int64_t, size_t> seenInputs;
    bool hasSingleDefault = false;
    size_t ruleIndex = 0;
    for (auto opaqueAttr : mappingAttr) {
        if (auto matchRuleAttr = llvm::dyn_cast<MatchRuleAttr>(opaqueAttr)) {
            if (matchRuleAttr.isDefault()) {
                if (hasSingleDefault) {
                    return emitOpError(
                        "Match operator has more than one default rule in its "
                        "mapping attribute.");
                } else {
                    hasSingleDefault = true;
                }
            }
            auto matchValues = matchRuleAttr.getValues();
            for (auto value : matchValues) {
                if (seenInputs.count(value) != 0) {
                    return emitOpError(
                               " has a duplicate input in its mapping "
                               "attribute.")
                           << " Input " << value << " in rule #" << ruleIndex
                           << ". Previously seen in rule #"
                           << seenInputs[value];
                }
                seenInputs.emplace(value, ruleIndex);
            }

            auto matchResult = matchRuleAttr.getIndex();
            if (matchResult >= numOptions) {
                return emitOpError(
                           " has a result index that is out of bounds in its "
                           "mapping attribute.")
                       << " Result index: " << matchResult
                       << " Number of options: " << numOptions;
            }
            ++ruleIndex;
        } else {
            return emitOpError(
                "Match operator has a non-MatchRuleAttr attribute in its "
                "mapping attribute.");
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// RVSDGDialect
//===----------------------------------------------------------------------===//

void RVSDGDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.cpp.inc"
        >();
}
