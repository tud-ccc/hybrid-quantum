// RUN: quantum-opt -convert-quantum-to-qir -convert-scf-to-cf -convert-qir-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -one-shot-bufferize="allow-unknown-ops" -finalize-memref-to-llvm -convert-index-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts %s -o %t.mlir
// RUN: quantum-translate -mlir-to-llvmir %t.mlir -o %t.ll
// RUN: just qir %t.ll -o %t.out
// RUN: %t.out || true
// RUN: sh -c '%t.out; echo $?' |  FileCheck %s

module {
  // Function to allocate a qubit, apply an X gate, measure and read the result
  func.func @main() -> i1  {
    // Allocate a single qubit in computational basis state 0
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // Apply an X gate to the qubit
    %q1 = "quantum.X" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // Measure the qubit and store the result
    %mt, %q_m = "quantum.measure" (%q1) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    "quantum.deallocate"(%q_m) : (!quantum.qubit<1>) -> ()
    %i = "index.constant" () {value = 0 : index} : () -> (index)
    %m = "tensor.extract" (%mt, %i) : (tensor<1xi1>, index) -> (i1)
    // Return the output value 1
    return %m : i1
  }
}

// CHECK: {{[1]}}

