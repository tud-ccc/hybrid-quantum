#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQUANTUMTOQIR
#include "cinm-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

struct QuantumToQirQubitTypeMapping;

void populateConvertQuantumToQIRPatterns(
    TypeConverter &typeConverter,
    QuantumToQirQubitTypeMapping &mapping,
    RewritePatternSet &patterns);

} // namespace quantum

std::unique_ptr<Pass> createConvertQuantumToQIRPass();

} // namespace mlir
