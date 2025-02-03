/// Main entry point for the cinm-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)
/// @author      Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"
#include "cinm-mlir/Dialect/Quantum/Transforms/Passes.h"
#include "cinm-mlir/Conversion/QuantumPasses.h"
#include "cinm-mlir/Dialect/QIR/IR/QIRDialect.h"
#include "cinm-mlir/Dialect/QIR/Transforms/Passes.h"
#include "cinm-mlir/Conversion/QIRPasses.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllExtensions.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;


int main(int argc, char *argv[]) {
  DialectRegistry registry;
  registry.insert<quantum::QuantumDialect>();
  registry.insert<qir::QIRDialect>();
  registerAllDialects(registry);
  registerAllPasses();
  registerAllExtensions(registry);
  registerQuantumConversionPasses();
  quantum::registerQuantumTransformsPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "quantum-mlir optimizer driver\n", registry));
}
