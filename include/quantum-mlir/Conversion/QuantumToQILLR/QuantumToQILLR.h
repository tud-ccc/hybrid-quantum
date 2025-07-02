#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQUANTUMTOQILLR
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

void populateConvertQuantumToQILLRPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace quantum

std::unique_ptr<Pass> createConvertQuantumToQILLRPass();

} // namespace mlir
