//===- TargetQASM.cpp - QIR to OpenQASM Translation -----------------------===//
//
// Translate QIR dialect ops into OpenQASM 2.0.
//
/// @file
/// @author     Washim Neupane (washim.neupane@outlook.com)
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Target/qasm/TargetQASM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <atomic>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir-c/Diagnostics.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <string>
using namespace mlir;
using namespace mlir::qir;

using llvm::formatv;

namespace {
class QASMEmitter {
public:
    QASMEmitter(raw_ostream &o) : os(o), names() {}

    LogicalResult emitOperation(Operation &op);

    /// Return the mapped name or creates a new mapping for value.
    std::string getOrCreateName(Value value);

    /// Returns the output stream.
    raw_ostream &ostream() { return os; };

private:
    raw_ostream &os;
    std::atomic_int next{0};
    llvm::DenseMap<Value, std::string> names;
};
} // namespace

static std::string printOperand(Value v)
{
    auto c = v.getDefiningOp<arith::ConstantOp>();
    auto result = mlir::cast<FloatAttr>(c.getValue());
    if (result)
        return std::to_string(result.getValueAsDouble());
    else
        emitError(v.getLoc(), "Defining op not a constant.");

    return "";
}

/// Emit the QASM header
static LogicalResult printHeader(QASMEmitter &emitter)
{
    raw_ostream &os = emitter.ostream();
    os << "OPENQASM 2.0;\n"
          "include \"qelib1.inc\";\n\n";
    return success();
}

static LogicalResult printQubitAlloc(QASMEmitter &emitter, AllocOp op)
{
    std::string name = emitter.getOrCreateName(op.getResult());
    raw_ostream &os = emitter.ostream();
    os << "qreg " << name << "[1];\n";
    return success();
}

static LogicalResult printResultAlloc(QASMEmitter &emitter, AllocResultOp op)
{
    std::string name = emitter.getOrCreateName(op.getResult());
    raw_ostream &os = emitter.ostream();
    os << "creg " << name << "[1];\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printPrimitiveGate(QASMEmitter &emitter, OpTy op, std::string opName)
{
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateName(op.getInput());
    os << opName << " " << name << ";\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printRotationGate(QASMEmitter &emitter, OpTy op, std::string opName)
{
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateName(op.getInput());
    std::string theta = printOperand(op.getAngle());
    os << opName << "(" << theta << ") " << name << ";\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printControledGate(QASMEmitter &emitter, OpTy op, std::string opname)
{
    raw_ostream &os = emitter.ostream();
    std::string control = emitter.getOrCreateName(op.getControl());
    std::string target = emitter.getOrCreateName(op.getTarget());
    os << opname << " " << control << ", " << target << ";\n";
    return success();
}

static LogicalResult printToffoli(QASMEmitter &emitter, CCXOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string control1 = emitter.getOrCreateName(op.getControl1());
    std::string control2 = emitter.getOrCreateName(op.getControl2());
    std::string target = emitter.getOrCreateName(op.getTarget());
    os << "ccx"
       << " " << control1 << ", " << control2 << ", " << target << ";\n";
    return success();
}

static LogicalResult printSwap(QASMEmitter &emitter, SwapOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string lhs = emitter.getOrCreateName(op.getLhs());
    std::string rhs = emitter.getOrCreateName(op.getRhs());
    os << "swap"
       << " " << lhs << ", " << rhs << ";\n";
    return success();
}

static LogicalResult printMeasure(QASMEmitter &emitter, MeasureOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string input = emitter.getOrCreateName(op.getInput());
    std::string result = emitter.getOrCreateName(op.getResult());
    os << "measure"
       << " " << input << " -> " << result << "[0];\n";
    return success();
}

static LogicalResult printBarrier(QASMEmitter &emitter, BarrierOp op)
{
    raw_ostream &os = emitter.ostream();
    auto operands = op.getOperands();
    os << "barrier ";
    for (auto it = operands.begin(); it != operands.end(); ++it) {
        if (it != operands.begin()) os << ", ";
        os << emitter.getOrCreateName(*it);
    }
    os << ";\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printControledRotationGate(QASMEmitter &emitter, OpTy op, std::string opname)
{
    raw_ostream &os = emitter.ostream();
    std::string control = emitter.getOrCreateName(op.getControl());
    std::string target = emitter.getOrCreateName(op.getTarget());
    std::string theta = printOperand(op.getAngle());
    os << opname << "(" << theta << ") " << control << ", " << target << ";\n";
    return success();
}

static LogicalResult printU3(QASMEmitter &emitter, U3Op op)
{
    std::string t = printOperand(op.getTheta());
    std::string p = printOperand(op.getPhi());
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateName(op.getInput());
    os << "u3(" << t << "," << p << "," << l << ") " << name << ";\n";
    return success();
}

static LogicalResult printU2(QASMEmitter &emitter, U2Op op)
{
    std::string p = printOperand(op.getPhi());
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateName(op.getInput());
    os << "u2(" << p << "," << l << ") " << name << ";\n";
    return success();
}

static LogicalResult printU1(QASMEmitter &emitter, U1Op op)
{
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateName(op.getInput());
    os << "u1(" << l << ") " << name << ";\n";
    return success();
}

std::string QASMEmitter::getOrCreateName(Value value)
{
    if (names.contains(value)) return names[value];

    std::string name = "reg" + std::to_string(next++);
    names.insert({value, name});
    return name;
}

LogicalResult QASMEmitter::emitOperation(Operation &op)
{
    LogicalResult result =
        TypeSwitch<Operation*, LogicalResult>(&op)
            // Memory allocations
            .Case<ModuleOp>(
                [&]([[maybe_unused]] ModuleOp m) { return printHeader(*this); })
            .Case<AllocOp>([&](AllocOp a) { return printQubitAlloc(*this, a); })
            .Case<AllocResultOp>(
                [&](AllocResultOp r) { return printResultAlloc(*this, r); })
            // Single-qubit gates
            .Case<HOp>(
                [&](HOp h) { return printPrimitiveGate<HOp>(*this, h, "h"); })
            .Case<XOp>(
                [&](XOp x) { return printPrimitiveGate<XOp>(*this, x, "x"); })
            .Case<YOp>(
                [&](YOp y) { return printPrimitiveGate<YOp>(*this, y, "y"); })
            .Case<ZOp>(
                [&](ZOp z) { return printPrimitiveGate<ZOp>(*this, z, "z"); })
            .Case<SOp>(
                [&](SOp s) { return printPrimitiveGate<SOp>(*this, s, "s"); })
            .Case<SdgOp>([&](SdgOp sdg) {
                return printPrimitiveGate<SdgOp>(*this, sdg, "sdg");
            })
            .Case<TOp>(
                [&](TOp t) { return printPrimitiveGate<TOp>(*this, t, "t"); })
            .Case<TdgOp>([&](TdgOp tdg) {
                return printPrimitiveGate<TdgOp>(*this, tdg, "tdg");
            })
            // controlled gates
            .Case<CNOTOp>([&](CNOTOp cx) {
                return printControledGate<CNOTOp>(*this, cx, "cx");
            })
            .Case<CZOp>([&](CZOp cz) {
                return printControledGate<CZOp>(*this, cz, "cz");
            })
            .Case<CCXOp>([&](CCXOp ccx) { return printToffoli(*this, ccx); })
            // Controled rotation gates
            .Case<CRzOp>([&](CRzOp crz) {
                return printControledRotationGate<CRzOp>(*this, crz, "crz");
            })
            .Case<CRyOp>([&](CRyOp cry) {
                return printControledRotationGate<CRyOp>(*this, cry, "cry");
            })
            // U1/U2/U3 gates
            .Case<U3Op>([&](U3Op u3) { return printU3(*this, u3); })
            .Case<U2Op>([&](U2Op u2) { return printU2(*this, u2); })
            .Case<U1Op>([&](U1Op u1) { return printU1(*this, u1); })
            // Rx/Ry/Rz gates
            .Case<RxOp>([&](RxOp rx) {
                return printRotationGate<RxOp>(*this, rx, "rx");
            })
            .Case<RyOp>([&](RyOp ry) {
                return printRotationGate<RyOp>(*this, ry, "ry");
            })
            .Case<RzOp>([&](RzOp rz) {
                return printRotationGate<RzOp>(*this, rz, "rz");
            })
            // Others
            .Case<BarrierOp>(
                [&](BarrierOp barrier) { return printBarrier(*this, barrier); })
            .Case<SwapOp>([&](SwapOp swap) { return printSwap(*this, swap); })
            .Case<MeasureOp>(
                [&](MeasureOp measure) { return printMeasure(*this, measure); })
            .Case<ResetOp>([&](ResetOp reset) {
                return printPrimitiveGate<ResetOp>(*this, reset, "reset");
            })
            // Ignored ops
            .Case<arith::ConstantOp>([](Operation*) { return success(); })
            .Case<ReadMeasurementOp>([](Operation*) { return success(); })
            // Default = error case
            .Default([](Operation*) { return failure(); });

    if (failed(result)) return failure();
    return result;
}

LogicalResult qir::QIRTranslateToQASM(Operation* op, raw_ostream &os)
{
    QASMEmitter emitter(os);

    LogicalResult result = success();
    auto walk = op->walk<WalkOrder::PreOrder>([&](Operation* child) {
        if (failed(result = emitter.emitOperation(*child)))
            return WalkResult::interrupt();
        return WalkResult::advance();
    });
    if (walk.wasInterrupted())
        emitError(op->getLoc(), "Interrupt of QIR to QASM translation.");
    return result;
}
