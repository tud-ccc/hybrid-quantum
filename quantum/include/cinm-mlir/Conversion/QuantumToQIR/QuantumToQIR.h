#pragma once

#include "cinm-mlir/Dialect/Quantum/IR/QuantumDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::quantum {

void populateQuantumToQIRConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertQuantumToQIRPass();

} // namespace mlir::quantum
