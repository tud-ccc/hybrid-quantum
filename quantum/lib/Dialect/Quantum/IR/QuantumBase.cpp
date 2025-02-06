/// Implements the Quantum dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.h"

#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"

#define DEBUG_TYPE "quantum-base"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

} // namespace

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize()
{
    registerOps();
    registerTypes();
}