/// Main entry point for the quantum-mlir optimizer driver.
///
/// @file
/// @author      Lars Schütze (lars.schuetze@tu-dresden.de)

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "quantum-mlir/Conversion/Passes.h"
#include "quantum-mlir/Dialect/QILLR/Extensions/InlinerExtension.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"
#include "quantum-mlir/Dialect/QPU/IR/QPU.h"
#include "quantum-mlir/Dialect/QQT/IR/QQT.h"
#include "quantum-mlir/Dialect/QQT/Transforms/Passes.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDG.h"

#include <mlir/InitAllExtensions.h>

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<rvsdg::RVSDGDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<qpu::QPUDialect>();
    registry.insert<qillr::QILLRDialect>();
    registry.insert<qqt::QQTDialect>();

    registerAllPasses();
    // rvsdg::registerAllPasses();
    qpu::registerQPUPasses();
    quantum::registerQuantumPasses();
    quantum::registerConversionPasses();
    qillr::registerQILLRPasses();
    qqt::registerQQTPasses();
    rvsdg::registerRVSDGPasses();

    registerAllExtensions(registry);
    qillr::registerInlinerExtension(registry);

    return asMainReturnCode(
        MlirOptMain(argc, argv, "quantum-mlir optimizer driver\n", registry));
}
