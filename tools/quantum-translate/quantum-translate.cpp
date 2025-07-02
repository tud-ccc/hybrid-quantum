/// Main entry point for the quantum-mlir optimizer driver.
///
/// @file
/// @author     Washim Neupane (washim.neupane@outlook.com)
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "quantum-mlir/Target/qasm/TargetQASM.h"

using namespace mlir;

int main(int argc, char** argv)
{
    registerAllTranslations();
    qillr::registerQILLRToOpenQASMTranslation();
    return failed(
        mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
