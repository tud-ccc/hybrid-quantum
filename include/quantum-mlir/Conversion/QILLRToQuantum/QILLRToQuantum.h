#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQILLRTOQUANTUM
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace qqt {

void populateConvertQILLRToQuantumPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    IRMapping &mapping,
    DominanceInfo &domInfo);

} // namespace qqt

std::unique_ptr<Pass> createConvertQILLRToQuantumPass();

} // namespace mlir
