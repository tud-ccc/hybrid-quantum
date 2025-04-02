// RUN: quantum-opt %s \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:       convert-quantum-to-qir, \
// RUN:       func.func(convert-scf-to-cf), \
// RUN:       convert-qir-to-llvm, \
// RUN:       convert-func-to-llvm, \
// RUN:       convert-cf-to-llvm, \
// RUN:       convert-vector-to-llvm, \
// RUN:       one-shot-bufferize{allow-unknown-ops}, \
// RUN:       finalize-memref-to-llvm, \
// RUN:       convert-index-to-llvm, \
// RUN:       convert-arith-to-llvm, \
// RUN:       reconcile-unrealized-casts)" | \
// RUN: mlir-runner -e entry -entry-point-result=void \
// RUN:     --shared-libs=%qir_shlibs,%mlir_c_runner_utils | \
// RUN: FileCheck %s --match-full-lines

module {

  // Function to allocate a qubit, apply an X gate, measure and read the result
  func.func @test_0_X_returns_1() -> ()  {
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    %q1 = "quantum.X" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %mt, %q_m = "quantum.measure" (%q1) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    "quantum.deallocate"(%q_m) : (!quantum.qubit<1>) -> ()
    %i = "index.constant" () {value = 0 : index} : () -> (index)
    %m = "tensor.extract" (%mt, %i) : (tensor<1xi1>, index) -> (i1)
    vector.print %m : i1
    return
  }

  func.func @entry() {
    // CHECK: 1
    func.call @test_0_X_returns_1() : () -> ()
    
    return
  }
}

