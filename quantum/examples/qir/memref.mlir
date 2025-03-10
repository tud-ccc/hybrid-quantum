module {
  // Quantum function that applies Rx and Rz gates to qubits
  func.func @quantum_fn(%theta_memref: memref<2xf64>) -> i1 {
    // Allocate a qubit
    %q0 = "qir.alloc"() : () -> (!qir.qubit)
    %r0 = "qir.ralloc"() : () -> (!qir.result)

    // Load theta values from the memrefs
    %c0 = arith.constant 0: index
    %c1 = arith.constant 1: index
    %theta1 = memref.load %theta_memref[%c0] : memref<2xf64>
    %theta2 = memref.load %theta_memref[%c1] : memref<2xf64>
    
    // Apply Rx and Rz gates using the loaded theta values
    "qir.Rx"(%q0, %theta1) : (!qir.qubit, f64) -> ()
    "qir.Rz"(%q0, %theta2) : (!qir.qubit, f64) -> ()
    "qir.measure"(%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    %res = "qir.read_measurement"(%r0) : (!qir.result) -> i1
    return %res : i1
  }

  // Main function that passes theta values to the quantum function
  func.func @main() -> i1 {
    // Allocate memory for two values representing the theta angles
    %theta_memref = memref.alloc() : memref<2xf64>

    // Store values for theta1 and theta2
    %theta1 = arith.constant 1.2 : f64
    %theta2 = arith.constant 2.3 : f64
    %c0 = arith.constant 0: index
    %c1 = arith.constant 1: index
    memref.store %theta1, %theta_memref[%c0] : memref<2xf64>
    memref.store %theta2, %theta_memref[%c1] : memref<2xf64>

    // Call quantum function and pass theta values as memrefs
    %out_val = call @quantum_fn(%theta_memref) : (memref<2xf64>) -> i1
    return %out_val : i1
  }
}
