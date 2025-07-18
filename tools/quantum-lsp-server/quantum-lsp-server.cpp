/// Main entry point for the quantum-mlir MLIR language server.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"

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
    registry.insert<qpu::QPUDialect>();
    registry.insert<qillr::QILLRDialect>();
    registry.insert<rvsdg::RVSDGDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
