//===- TargetQASMRegistration.cpp - Register QIR to OpenQASM Translation --===//
//
// Registers the QIR to OpenQASM translation
//
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "quantum-mlir/Dialect/QIR/IR/QIRBase.h"
#include "quantum-mlir/Target/qasm/TargetQASM.h"

using namespace mlir;
using namespace mlir::qir;

//===----------------------------------------------------------------------===//
// QIR to QASM registration
//===----------------------------------------------------------------------===//

void mlir::qir::registerQIRToOpenQASMTranslation()
{
    TranslateFromMLIRRegistration registration(
        "mlir-to-openqasm",
        "Translate QIR dialect to OpenQASM 2.0",
        [](Operation* op, raw_ostream &os) -> LogicalResult {
            return qir::QIRTranslateToQASM(op, os);
        },
        [](DialectRegistry &registry) {
            registry.insert<qir::QIRDialect>();
            registry.insert<arith::ArithDialect>();
        });
}
