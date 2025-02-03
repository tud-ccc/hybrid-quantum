#pragma once

#include "cinm-mlir/Dialect/QIR/IR/QIRDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::qir {

void populateQIRToLLVMConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertQIRToLLVMPass();

} // namespace mlir::qir
