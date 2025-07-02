//===- TargetQASMRegistration.cpp - Register QILLR to OpenQASM Translation-===//
//
// Registers the QILLR to OpenQASM translation
//
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Target/qasm/TargetQASM.h"

using namespace mlir;
using namespace mlir::qillr;

//===----------------------------------------------------------------------===//
// QILLR to QASM registration
//===----------------------------------------------------------------------===//

void mlir::qillr::registerQILLRToOpenQASMTranslation()
{
    TranslateFromMLIRRegistration registration(
        "mlir-to-openqasm",
        "Translate QILLR dialect to OpenQASM 2.0",
        [](Operation* op, raw_ostream &os) -> LogicalResult {
            return qillr::QILLRTranslateToQASM(op, os);
        },
        [](DialectRegistry &registry) {
            registry.insert<qillr::QILLRDialect>();
            registry.insert<arith::ArithDialect>();
        });
}
