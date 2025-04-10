module {
  func.func @epr_pair() {
    // Allocate 2 qubits into a single quantum register.
    %q_init = "quantum.alloc"() : () -> (!quantum.qubit<2>)
    %q0_0, %q1_0 = "quantum.split"(%q_init) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // Apply a Hadamard gate to the first qubit.
    // Use CNOT with the first qubit (after H) as control and the second as target.
    %q0_h = "quantum.H"(%q0_0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_epr, %q1_epr = "quantum.CNOT"(%q0_h, %q1_0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // Measure both qubits to observe correlation.
    %m0, %q0_meas = "quantum.measure_single"(%q0_epr) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    %m1, %q1_meas = "quantum.measure_single"(%q1_epr) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "quantum.deallocate"(%q0_meas) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%q1_meas) : (!quantum.qubit<1>) -> ()
    vector.print %m0 : i1
    vector.print %m1 : i1

    return
  }

  func.func @entry() {
    // The printed outcomes should be correlated.
    func.call @epr_pair() : () -> ()
    return
  }
}
