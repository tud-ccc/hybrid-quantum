/// Implements the Quantum dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-ops"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
      >();
}

LogicalResult QuantumDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  if (!llvm::isa<UnitAttr>(attr.getValue()) ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';
  return success();
}


//Verfiers

LogicalResult XOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same type");
  return success();
}

LogicalResult CNOTOp::verify() {
  return success();
}

LogicalResult InsertOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null index";
    }
    return success();
}

LogicalResult ExtractOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null index";
    }
    return success();
}


// //Parsers
// ParseResult CircuitOp::parse(OpAsmParser &parser, OperationState &result) {
//   SmallVector<OpAsmParser::UnresolvedOperand, 4> qubitOperands;
//   Region &body = *result.addRegion();

//   // Parse input operands
//   if (parser.parseLParen() ||
//       parser.parseOperandList(qubitOperands) ||
//       parser.parseRParen())
//     return failure();

//   // Parse the body region
//   if (parser.parseRegion(body, /*arguments=*/{}, /*argTypes=*/{}))
//     return failure();

//   // Resolve operands
//   if (parser.resolveOperands(qubitOperands, parser.getBuilder().getType<qubitType>(), result.operands))
//     return failure();

//   // Infer result types from the terminator
//   if (!body.empty()) {
//     if (auto returnOp = dyn_cast<ReturnOp>(body.front().back())) {
//       result.types.assign(returnOp.getOperandTypes().begin(), returnOp.getOperandTypes().end());
//     }
//   }

//   return success();
// }

// // Custom printer
// void CircuitOp::print(OpAsmPrinter &p) {
//   p << "(";
//   llvm::interleaveComma(getInputs(), p, [&](Value input) {
//     p << input << " : " << input.getType();
//   });
//   p << ") ";
//   p.printRegion(getBody(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
// }
//===- Generated implementation -------------------------------------------===//
//===----------------------------------------------------------------------===//
// CircuitOp
//
// This code section was derived and modified from the LLVM project FuncOp
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

mlir::ParseResult CircuitOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void CircuitOp::print(mlir::OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

namespace {
/// Verify the argument list and entry block are in agreement.
LogicalResult verifyArgumentAndEntry_(CircuitOp op) {
  auto fnInputTypes = op.getFunctionType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i != entryBlock.getNumArguments(); ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';
  return success();
}

/// Verify that no classical values are created/used in the circuit outside of
/// values that originate as argument values or the result of a measurement.
LogicalResult verifyClassical_(CircuitOp op) {
  mlir::Operation *classicalOp = nullptr;
  WalkResult const result = op->walk([&](Operation *subOp) {
    if (isa<mlir::arith::ConstantOp>(subOp) ||
        isa<quantum::ReturnOp>(subOp) || 
        isa<CircuitOp>(subOp) ||
        subOp->hasTrait<mlir::OpTrait::Unitary>() ||
        subOp->hasTrait<mlir::OpTrait::Kernel>() ||
        isa<mlir::quantum::QuantumDialect>(subOp->getDialect()) ||
        isa<mlir::scf::SCFDialect>(subOp->getDialect())
        )
      return WalkResult::advance();
    classicalOp = subOp;
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted())
    return classicalOp->emitOpError()
           << "is classical and should not be inside a circuit.";
  return success();
}
} // anonymous namespace

LogicalResult CircuitOp::verify() {
  // If external will be linked in later and nothing to do
  if (isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(*this)))
    return mlir::failure();

  if (failed(verifyClassical_(*this)))
    return mlir::failure();

  return success();
}

CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type, ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  CircuitOp::build(builder, state, name, type, attrs);
  return cast<CircuitOp>(Operation::create(state));
}
CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type,
                            Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> const attrRef(attrs);
  return create(location, name, type, attrRef);
}
CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type, ArrayRef<NamedAttribute> attrs,
                            ArrayRef<DictionaryAttr> argAttrs) {
  CircuitOp circ = create(location, name, type, attrs);
  circ.setAllArgAttrs(argAttrs);
  return circ;
}

void CircuitOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      FunctionType type, ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

/// Clone the internal blocks and attributes from this circuit to the
/// destination circuit.
void CircuitOp::cloneInto(CircuitOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this circuit and all of its block.
/// Remap any operands that use values outside of the function
/// Using the provider mapper. Replace references to
/// cloned sub-values with the corresponding copied value and
/// add to the mapper
CircuitOp CircuitOp::clone(IRMapping &mapper) {
  FunctionType newType = getFunctionType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool const isExternalCircuit = isExternal();
  if (!isExternalCircuit) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0; i != getNumArguments(); ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(getContext(), inputTypes, newType.getResults());
  }

  // Create the new circuit
  CircuitOp newCirc = cast<CircuitOp>(getOperation()->cloneWithoutRegions());
  newCirc.setType(newType);

  // Clone the current function into the new one and return.
  cloneInto(newCirc, mapper);
  return newCirc;
}

CircuitOp CircuitOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
//
// end CircuitOp

//===----------------------------------------------------------------------===//
// ReturnOp
//
// This code section was derived and modified from the LLVM project's standard
// dialect ReturnOp.
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

LogicalResult mlir::quantum::ReturnOp::verify() {
  auto circuit = (*this)->getParentOfType<CircuitOp>();

  FunctionType const circuitType = circuit.getFunctionType();

  auto numResults = circuitType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != getOperands().size()) {
    return emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(circuit.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(circuitType.getResults(), getOperands())) {
    auto opType = operand.getType();
    if (type != opType) {
      return emitOpError() << "unexpected type `" << opType << "' for operand #"
                           << i;
    }
    i++;
  }
  return success();
}

//===----------------------------------------------------------------------===//
//
// end ReturnOp
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// CallCircuitOp
//===----------------------------------------------------------------------===//

auto CallCircuitOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult
CallCircuitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  auto circuitAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!circuitAttr)
    return emitOpError("Requires a 'callee' symbol reference attribute");

  auto circuit =
      symbolTable.lookupNearestSymbolFrom<CircuitOp>(*this, circuitAttr);
  if (!circuit)
    return emitOpError() << "'" << circuitAttr.getValue()
                         << "' does not reference a valid circuit";

  // Verify the types match
  auto circuitType = circuit.getFunctionType();
  if (circuitType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for the callee circuit");

  for (unsigned i = 0; i != circuitType.getNumInputs(); ++i) {
    if (getOperand(i).getType() != circuitType.getInput(i)) {
      auto diag = emitOpError("operand type mismatch at index ") << i;
      diag.attachNote() << "op input types: " << getOperandTypes();
      diag.attachNote() << "function operand types: "
                        << circuitType.getInputs();
      return diag;
    }
  }

  if (circuitType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for the callee circuit");

  for (unsigned i = 0; i != circuitType.getNumResults(); ++i) {
    if (getResult(i).getType() != circuitType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "op result types: " << getResultTypes();
      diag.attachNote() << "function result types: "
                        << circuitType.getResults();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
//
// end CallCircuitOp
//
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"