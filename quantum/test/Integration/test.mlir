// RUN: quantum-opt --convert-quantum-to-qir --convert-scf-to-cf --canonicalize --finalize-memref-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --convert-qir-to-llvm --reconcile-unrealized-casts  %s -o %t.mlir
// RUN: quantum-translate --mlir-to-llvmir %t.mlir -o %t.ll
// RUN: just qir %t.ll -o %t.out
// RUN: %t.out || true
// RUN: sh -c '%t.out; echo $?' |  FileCheck %s

module {
  // Function to allocate a qubit, apply an X gate, measure and read the result
  func.func @main() -> i1  {
    // Allocate a single qubit
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    
    // Apply an X gate to the qubit
    %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Measure the qubit and store the result
    %m, %q_m = "quantum.measure" (%q1) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)

    // Return the output value 0 or 1 with 50-50 probability.  
    return %m : i1
  }
}

// CHECK: {{[01]}}

