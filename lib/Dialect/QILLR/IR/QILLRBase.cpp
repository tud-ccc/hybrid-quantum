/// Implements the QILLR dialect base.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/tensor/IR/Tensor.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"

using namespace mlir;
using namespace mlir::qillr;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// QILLRDialect
//===----------------------------------------------------------------------===//

void QILLRDialect::initialize()
{
    registerOps();
    registerTypes();
}
