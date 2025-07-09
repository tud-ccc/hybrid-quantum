#ifndef LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_
#define LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace qpu {

class SabreSwapAnalysis {

public:
    SabreSwapAnalysis(Operation* op);
    ~SabreSwapAnalysis() = default;

    /// Return true if a `quantum.swap` op should be inserted after the given
    /// operation, according to the solution to the optimization problem.
    bool shouldInsertSwap(Operation* op) const { return solution.lookup(op); }

private:
    llvm::DenseMap<Operation*, bool> solution;
};

} // namespace qpu
} // namespace mlir

#endif // LIB_ANALYSIS_SABRESWAPANALYSIS_SABRESWAPANALYSIS_H_
