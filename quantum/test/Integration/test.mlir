// RUN: quantum-opt --convert-arith-to-llvm --convert-func-to-llvm --convert-qir-to-llvm %s -o %t.mlir
// RUN: quantum-translate --mlir-to-llvmir %t.mlir -o %t.ll
// RUN: just qir %t.ll -o %t.out
// RUN: %t.out 
// RUN: sh -c '%t.out; echo Exit code: $?'| FileCheck %s

module {
  func.func @quantum_operation() {
    %q0 = "qir.alloc"() : () -> (!qir.qubit)
    %r0 = "qir.ralloc"() : () -> (!qir.result)
    
    // Apply X gate
    "qir.X"(%q0) : (!qir.qubit) -> ()
    
    // Measure
    "qir.measure"(%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    
    // Read and print measurement result
    %result = "qir.read_measurement"(%r0) : (!qir.result) -> (i1)
    
    return
  }

  func.func @main() -> i32 {
    call @quantum_operation() : () -> ()
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}

// CHECK: OUTPUT RESULT 1

