//===- qir2qasm.cpp - QIR → OpenQASM (AllocOp only) ---------------------===//
//
// Minimal translator: each QIR AllocOp becomes a single-qubit qreg.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "quantum-mlir/Dialect/QIR/IR/QIR.h"
#include "quantum-mlir/Dialect/QIR/IR/QIROps.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace qir;

namespace {
struct QASMTranslation {
    ModuleOp module;
    raw_ostream &os;
    unsigned idx = 0;

    QASMTranslation(ModuleOp m, raw_ostream &o) : module(m), os(o) {}

    LogicalResult translate()
    {
        // OpenQASM header
        os << "OPENQASM 2.0;\n"
           << "include \"qelib1.inc\";\n\n";

        // For each QIR AllocOp, emit one qreg
        for (auto allocOp : module.getOps<AllocOp>())
            os << "qreg q" << idx++ << "[1];\n";
        return success();
    }
};
} // end anonymous namespace

namespace mlir {
/// Register under --mlir-to-openqasm
void registerToOpenQASMTranslation()
{
    static TranslateFromMLIRRegistration reg(
        /*option name*/ "mlir-to-openqasm",
        /*help text*/ "Translate QIR dialect to OpenQASM 2.0",
        /*translate fn*/
        [](ModuleOp m, raw_ostream &os) -> LogicalResult {
            return QASMTranslation(m, os).translate();
        },
        /*dialects reg*/
        [](DialectRegistry &registry) { registry.insert<qir::QIRDialect>(); });
}
} // namespace mlir
