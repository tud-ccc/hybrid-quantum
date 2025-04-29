#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQIRTOQUANTUM
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qir {

void populateConvertQIRToQuantumPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace qir

std::unique_ptr<Pass> createConvertQIRToQuantumPass();

} // namespace mlir
