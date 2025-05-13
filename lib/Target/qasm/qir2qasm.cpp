//===- qir2qasm.cpp - QIR → OpenQASM Translation ------------------------===//
//
// A small, maintainable translator that handles AllocOp, HOp, and all
// other relevant QIR dialect ops, flat-dispatch via TypeSwitch,
// and per-op printer methods for OpenQASM 2.0.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h" // for arith::ConstantOp, IntegerAttr, FloatAttr
#include "mlir/IR/BuiltinOps.h"          // ModuleOp
#include "mlir/Support/LogicalResult.h" // LogicalResult, success(), failure()
#include "mlir/Tools/mlir-translate/Translation.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h" // All QIR ops: AllocOp, HOp, XOp, ZOp, etc.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h" // TypeSwitch
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace qir;

namespace {
class QASMTranslation {
public:
    QASMTranslation(ModuleOp m, raw_ostream &o) : module(m), os(o) {}

    LogicalResult translate()
    {
        emitHeader();
        auto result = module.walk<WalkOrder::PreOrder>([&](Operation* op) {
            return TypeSwitch<Operation*, WalkResult>(op)
                // Initialization / Diagnostics
                .Case<InitOp>([&](InitOp i) { return handleInit(i); })
                .Case<SeedOp>([&](SeedOp s) { return handleSeed(s); })
                .Case<ShowStateOp>(
                    [&](ShowStateOp s) { return handleShowState(s); })

                // Memory allocations
                .Case<AllocOp>([&](AllocOp a) { return handleQubitAlloc(a); })
                .Case<AllocResultOp>(
                    [&](AllocResultOp r) { return handleCRegAlloc(r); })

                // Single-qubit gates
                .Case<HOp>([&](HOp h) { return handleH(h); })
                .Case<XOp>([&](XOp x) { return handleX(x); })
                .Case<ZOp>([&](ZOp z) { return handleZ(z); })
                .Case<YOp>([&](YOp y) { return handleY(y); })
                .Case<RxOp>([&](RxOp r) { return handleRx(r); })
                .Case<RzOp>([&](RzOp r) { return handleRz(r); })
                .Case<RyOp>([&](RyOp r) { return handleRy(r); })
                .Case<U3Op>([&](U3Op u) { return handleU3(u); })
                .Case<U2Op>([&](U2Op u) { return handleU2(u); })
                .Case<U1Op>([&](U1Op u) { return handleU1(u); })
                .Case<SOp>([&](SOp s) { return handleS(s); })
                .Case<SdgOp>([&](SdgOp s) { return handleSdg(s); })
                .Case<TOp>([&](TOp t) { return handleT(t); })
                .Case<TdgOp>([&](TdgOp t) { return handleTdg(t); })

                // Multi-qubit gates
                .Case<CNOTOp>([&](CNOTOp c) { return handleCNOT(c); })
                .Case<CZOp>([&](CZOp c) { return handleCZ(c); })
                .Case<CRzOp>([&](CRzOp c) { return handleCRz(c); })
                .Case<CRyOp>([&](CRyOp c) { return handleCRy(c); })
                .Case<CCXOp>([&](CCXOp c) { return handleCCX(c); })
                .Case<BarrierOp>([&](BarrierOp b) { return handleBarrier(b); })
                .Case<SwapOp>([&](SwapOp s) { return handleSwap(s); })

                // Measurement / Reset
                .Case<MeasureOp>([&](MeasureOp m) { return handleMeasure(m); })
                .Case<ReadMeasurementOp>([&](ReadMeasurementOp r) {
                    return handleReadMeasurement(r);
                })
                .Case<ResetOp>([&](ResetOp r) { return handleReset(r); })

                .Default([&](Operation*) { return WalkResult::advance(); });
        });
        return result.wasInterrupted() ? failure() : success();
    }

private:
    ModuleOp module;
    raw_ostream &os;
    unsigned nextQ = 0, nextC = 0;
    llvm::DenseMap<Value, std::string> qubitNames, cregNames;

    // Emit the QASM header
    void emitHeader()
    {
        os << "OPENQASM 2.0;\n"
              "include \"qelib1.inc\";\n\n";
    }

    //===----------------------------------------------------------------------===//
    // Handlers for each op
    //===----------------------------------------------------------------------===//

    // init → (no QASM equivalent)
    WalkResult handleInit(InitOp op)
    {
        os << "// init\n";
        return WalkResult::advance();
    }

    // seed → comment
    WalkResult handleSeed(SeedOp op)
    {
        auto c = op.getSeed().getDefiningOp<arith::ConstantOp>();
        int64_t v = c.getValue().cast<IntegerAttr>().getInt();
        os << "// seed " << v << "\n";
        return WalkResult::advance();
    }

    // show_state → comment
    WalkResult handleShowState(ShowStateOp op)
    {
        os << "// show_state\n";
        return WalkResult::advance();
    }

    // qreg qN[1];
    WalkResult handleQubitAlloc(AllocOp op)
    {
        std::string name = "q" + std::to_string(nextQ++);
        qubitNames[op.getResult()] = name;
        os << "qreg " << name << "[1];\n";
        return WalkResult::advance();
    }

    // creg cN[1];
    WalkResult handleCRegAlloc(AllocResultOp op)
    {
        std::string name = "c" + std::to_string(nextC++);
        cregNames[op.getResult()] = name;
        os << "creg " << name << "[1];\n";
        return WalkResult::advance();
    }

    // h q;
    WalkResult handleH(HOp op)
    {
        os << "h " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // x q;
    WalkResult handleX(XOp op)
    {
        os << "x " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // z q;
    WalkResult handleZ(ZOp op)
    {
        os << "z " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // y q;
    WalkResult handleY(YOp op)
    {
        os << "y " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // rx(angle) q;
    WalkResult handleRx(RxOp op)
    {
        double angle = getAngle(op.getOperand(1));
        os << "rx(" << angle << ") " << qubitNames.lookup(op.getOperand(0))
           << ";\n";
        return WalkResult::advance();
    }

    // rz(angle) q;
    WalkResult handleRz(RzOp op)
    {
        double angle = getAngle(op.getOperand(1));
        os << "rz(" << angle << ") " << qubitNames.lookup(op.getOperand(0))
           << ";\n";
        return WalkResult::advance();
    }

    // ry(angle) q;
    WalkResult handleRy(RyOp op)
    {
        double angle = getAngle(op.getOperand(1));
        os << "ry(" << angle << ") " << qubitNames.lookup(op.getOperand(0))
           << ";\n";
        return WalkResult::advance();
    }

    // U3(theta,phi,lambda) q;
    WalkResult handleU3(U3Op op)
    {
        auto getParam = [&](Value v) {
            auto c = v.getDefiningOp<arith::ConstantOp>();
            return c.getValue().cast<FloatAttr>().getValueAsDouble();
        };
        double t = getParam(op.getOperand(1));
        double p = getParam(op.getOperand(2));
        double l = getParam(op.getOperand(3));
        os << "u3(" << t << "," << p << "," << l << ") "
           << qubitNames.lookup(op.getOperand(0)) << ";\n";
        return WalkResult::advance();
    }

    // u2(phi,lambda) q;
    WalkResult handleU2(U2Op op)
    {
        double p = getAngle(op.getOperand(1));
        double l = getAngle(op.getOperand(2));
        os << "u2(" << p << "," << l << ") "
           << qubitNames.lookup(op.getOperand(0)) << ";\n";
        return WalkResult::advance();
    }

    // u1(lambda) q;
    WalkResult handleU1(U1Op op)
    {
        double l = getAngle(op.getOperand(1));
        os << "u1(" << l << ") " << qubitNames.lookup(op.getOperand(0))
           << ";\n";
        return WalkResult::advance();
    }

    // s q;
    WalkResult handleS(SOp op)
    {
        os << "s " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // sdg q;
    WalkResult handleSdg(SdgOp op)
    {
        os << "sdg " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // t q;
    WalkResult handleT(TOp op)
    {
        os << "t " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // tdg q;
    WalkResult handleTdg(TdgOp op)
    {
        os << "tdg " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    // cx control,target;
    WalkResult handleCNOT(CNOTOp op)
    {
        os << "cx " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ";\n";
        return WalkResult::advance();
    }

    // cz control,target;
    WalkResult handleCZ(CZOp op)
    {
        os << "cz " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ";\n";
        return WalkResult::advance();
    }

    // crz(angle) control,target;
    WalkResult handleCRz(CRzOp op)
    {
        double a = getAngle(op.getOperand(2));
        os << "crz(" << a << ") " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ";\n";
        return WalkResult::advance();
    }

    // cry(angle) control,target;
    WalkResult handleCRy(CRyOp op)
    {
        double a = getAngle(op.getOperand(2));
        os << "cry(" << a << ") " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ";\n";
        return WalkResult::advance();
    }

    // ccx c1,c2,target;
    WalkResult handleCCX(CCXOp op)
    {
        os << "ccx " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ", "
           << qubitNames.lookup(op.getOperand(2)) << ";\n";
        return WalkResult::advance();
    }

    // barrier q0,q1,...;
    WalkResult handleBarrier(BarrierOp op)
    {
        os << "barrier ";
        for (auto it = op.getOperands().begin(); it != op.getOperands().end();
             ++it) {
            if (it != op.getOperands().begin()) os << ", ";
            os << qubitNames.lookup(*it);
        }
        os << ";\n";
        return WalkResult::advance();
    }

    // swap q0,q1;
    WalkResult handleSwap(SwapOp op)
    {
        os << "swap " << qubitNames.lookup(op.getOperand(0)) << ", "
           << qubitNames.lookup(op.getOperand(1)) << ";\n";
        return WalkResult::advance();
    }

    // measure q -> c[0];
    WalkResult handleMeasure(MeasureOp op)
    {
        os << "measure " << qubitNames.lookup(op.getOperand(0)) << " -> "
           << cregNames.lookup(op.getOperand(1)) << "[0];\n";
        return WalkResult::advance();
    }

    // read_measurement → comment
    WalkResult handleReadMeasurement(ReadMeasurementOp op)
    {
        os << "// read_measurement into " << cregNames.lookup(op.getOperand())
           << "\n";
        return WalkResult::advance();
    }

    // reset q;
    WalkResult handleReset(ResetOp op)
    {
        os << "reset " << qubitNames.lookup(op.getOperand()) << ";\n";
        return WalkResult::advance();
    }

    //===----------------------------------------------------------------------===//
    // Utilities
    //===----------------------------------------------------------------------===//
    double getAngle(Value v)
    {
        auto c = v.getDefiningOp<arith::ConstantOp>();
        assert(c && "Parameter must be constant");
        return c.getValue().cast<FloatAttr>().getValueAsDouble();
    }
};

} // end anonymous namespace

namespace mlir {
void registerToOpenQASMTranslation()
{
    static TranslateFromMLIRRegistration reg(
        "mlir-to-openqasm",
        "Translate QIR dialect to OpenQASM 2.0",
        [](ModuleOp m, raw_ostream &os) -> LogicalResult {
            return QASMTranslation(m, os).translate();
        },
        [](DialectRegistry &dr) {
            dr.insert<qir::QIRDialect, arith::ArithDialect>();
        });
}
} // namespace mlir
