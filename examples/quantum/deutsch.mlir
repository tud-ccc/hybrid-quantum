module {
  // Deutsch algorithm using a balanced oracle (f(x)=x).
  func.func @deutsch() {
    // Allocate 2 qubits: first qubit for the query, second for the ancilla.
    %q_init = "quantum.alloc"() : () -> (!quantum.qubit<2>)
    %q0_0, %q1_0 = "quantum.split"(%q_init) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // Initialize the ancilla to |1⟩ (first qubit remains |0⟩).
    %q1_1 = "quantum.X"(%q1_0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Apply Hadamard gates to both qubits.
    %q0_h = "quantum.H"(%q0_0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_h = "quantum.H"(%q1_1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q_h = "quantum.merge"(%q0_h, %q1_h) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)

    // Oracle: Implement the balanced function f(x)=x via CNOT.
    %q0_oracle, %q1_oracle = "quantum.split"(%q_h) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q0_u, %q1_u = "quantum.CNOT"(%q0_oracle, %q1_oracle) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q_postOracle = "quantum.merge"(%q0_u, %q1_u) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)

    // Apply Hadamard to the first qubit to complete interference. Leave the second qubit unchanged.
    %q0_split, %q1_split = "quantum.split"(%q_postOracle) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q0_final = "quantum.H"(%q0_split) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q_final = "quantum.merge"(%q0_final, %q1_split) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)

    // Measure the first qubit using quantum.measure_single.
    %m, %q0_measured = "quantum.measure_single"(%q0_final) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "quantum.deallocate"(%q0_measured) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%q1_split) : (!quantum.qubit<1>) -> ()

    // Print the measurement result.
    vector.print %m : i1
    return
  }

  func.func @entry() {
    // For a balanced function, the Deutsch algorithm predicts output 1.
    func.call @deutsch() : () -> ()
    return
  }
}
