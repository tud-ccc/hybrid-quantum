/// Main entry point for the cinm-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)
/// @author      Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/Passes.h"
#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;


int main(int argc, char *argv[]) {
  DialectRegistry registry;
  registerAllDialects(registry);

  registry.insert<quantum::QuantumDialect>();
  registry.insert<qir::QIRDialect>();
  
  registerAllPasses();
  registerTransformsPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "quantum-mlir optimizer driver\n", registry));
}
