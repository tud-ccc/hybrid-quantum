//Example code to demonstrate quantum teleportation
module {
  func.func @main() -> (i1, i1) {
    // Allocate three qubits: 
    // q0 holds the state to teleport, q1 and q2 form an entangled pair.
    %q0 = "qir.alloc"() : () -> (!qir.qubit)
    %q1 = "qir.alloc"() : () -> (!qir.qubit)
    %q2 = "qir.alloc"() : () -> (!qir.qubit)

    // Entangle q1 and q2 to create a Bell pair.
    "qir.H"(%q1) : (!qir.qubit) -> ()
    "qir.CNOT"(%q1, %q2) : (!qir.qubit, !qir.qubit) -> ()

    // --- Teleportation Protocol ---
    // 1. Prepare the Bell measurement between the qubit to teleport (q0)
    //    and one half of the entangled pair (q1).
    "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
    "qir.H"(%q0) : (!qir.qubit) -> ()

    // 2. Measure q0 and q1 to obtain classical bits m0 and m1.
    %r0 = "qir.ralloc"() : () -> (!qir.result)
    "qir.measure"(%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    %m0_tensor = "qir.read_measurement"(%r0) : (!qir.result) -> tensor<1xi1>
    
    %r1 = "qir.ralloc"() : () -> (!qir.result)
    "qir.measure"(%q1, %r1) : (!qir.qubit, !qir.result) -> ()
    %m1_tensor = "qir.read_measurement"(%r1) : (!qir.result) -> tensor<1xi1>

    // Create an index constant for extraction
    %c0 = arith.constant 0 : index
    %m0 = tensor.extract %m0_tensor[%c0] : tensor<1xi1>
    %m1 = tensor.extract %m1_tensor[%c0] : tensor<1xi1>

    // 3. Based on the measurement outcomes, correct the state on q2.
    // If m1 is true, apply an X gate.
    scf.if %m1 {
      "qir.X"(%q2) : (!qir.qubit) -> () 
    }
    
    // If m0 is true, apply a Z gate.
    scf.if %m0 {
      "qir.Z"(%q2) : (!qir.qubit) -> ()
    }

    // Measure the corrected qubit (q2_final) to verify teleportation.
    %r2 = "qir.ralloc"() : () -> (!qir.result)
    "qir.measure"(%q2, %r2) : (!qir.qubit, !qir.result) -> ()
    %final_tensor = "qir.read_measurement"(%r2) : (!qir.result) -> tensor<1xi1>
    %final = tensor.extract %final_tensor[%c0] : tensor<1xi1>

    // Return two classical bits for demonstration:
    // Here we return the measurement outcome m0 (from q0) and the final state from q2.
    return %m0, %final : i1, i1
  }
}
