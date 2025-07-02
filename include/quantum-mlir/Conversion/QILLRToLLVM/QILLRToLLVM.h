/// Declaration of the QILLR to LLVM conversion pass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQILLRTOLLVM
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qillr {

struct AllocationAnalysis;

void populateConvertQILLRToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    AllocationAnalysis &analysis);

} // namespace qillr

/// Constructs the convert-qillr-to-llvm pass.
std::unique_ptr<Pass> createConvertQILLRToLLVMPass();

} // namespace mlir
