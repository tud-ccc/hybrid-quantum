#pragma once

#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::quantum {

void populateQuantumToLLVMConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertQuantumToLLVMPass();

} // namespace mlir::quantum
