/// Declaration of the QIR to LLVM conversion pass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQIRTOLLVM
#include "cinm-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qir {

void populateConvertQIRToLLVMPatterns (
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace qir

/// Constructs the convert-qir-to-llvm pass.
std::unique_ptr<Pass> createConvertQIRToLLVMPass();

} // namespace mlir