/// Main entry point for the quantum-mlir optimizer driver.
///
/// @file

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "quantum-mlir/Target/qasm/qir2qasm.h"
using namespace mlir;

int main(int argc, char** argv)
{
    registerAllTranslations();
    registerToOpenQASMTranslation();
    return failed(
        mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
