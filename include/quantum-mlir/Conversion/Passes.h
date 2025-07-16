/// Declaration of the conversion pass within Quantum dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#pragma once

#include "quantum-mlir/Conversion/QILLRToLLVM/QILLRToLLVM.h"
#include "quantum-mlir/Conversion/QILLRToQuantum/QILLRToQuantum.h"
#include "quantum-mlir/Conversion/QuantumToQILLR/QuantumToQILLR.h"
#include "quantum-mlir/Conversion/SCFToRVSDG/ScfToRVSDG.h"

namespace mlir::quantum {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::quantum
