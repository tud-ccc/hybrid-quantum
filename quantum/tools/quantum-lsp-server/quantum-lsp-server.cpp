/// Main entry point for the quantum-mlir MLIR language server.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "cinm-mlir/Dialect/QIR/IR/QIRBase.h"
#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<quantum::QuantumDialect>();
    registry.insert<qir::QIRDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}