module {
  func.func @teleport() {
    // Allocate 3 qubits:
    //   - Qubit A: the state to teleport.
    //   - Qubits B & C: used to create the EPR pair.
    %q_all = "quantum.alloc"() : () -> (!quantum.qubit<3>)
    %qA, %qBC = "quantum.split"(%q_all) : (!quantum.qubit<3>) -> (!quantum.qubit<1>, !quantum.qubit<2>)
    %qB, %qC = "quantum.split"(%qBC) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // --- Preparation ---
    // Prepare the state to teleport on qubit A.
    %qA_prep = "quantum.H"(%qA) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Create an EPR pair between qubits B and C.
    %qB_h = "quantum.H"(%qB) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %qB_epr, %qC_epr = "quantum.CNOT"(%qB_h, %qC) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // --- Bell Measurement on Alice's side ---
    // Alice holds qubits A_prep and qB_epr.
    // Step 1: Apply CNOT with qubit A as control and qubit B as target.
    %qA_bell, %qB_bell = "quantum.CNOT"(%qA_prep, %qB_epr) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    // Step 2: Apply Hadamard to qubit A.
    %qA_bell_h = "quantum.H"(%qA_bell) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Measure qubits A and B (Bell measurement outcomes).
    %mA, %qA_meas = "quantum.measure_single"(%qA_bell_h) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    %mB, %qB_meas = "quantum.measure_single"(%qB_bell) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)

    // --- Classical Correction on Bob's qubit ---
    // Correction 1: If mB is 1, apply X gate on Bob's qubit.
    %qC_afterX = quantum.if %mB qubits(%qC_epr_in = %qC_epr) -> (!quantum.qubit<1>) {
      %qC_corr = "quantum.X"(%qC_epr_in) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
      "quantum.yield" (%qC_corr) : (!quantum.qubit<1>) -> ()
    } else {
      "quantum.yield" (%qC_epr_in) : (!quantum.qubit<1>) -> ()
    }

    // Correction 2: If mA is 1, apply Z gate on Bob's qubit.
    %qC_corrected = quantum.if %mA qubits(%qC_afterX_in = %qC_afterX) -> (!quantum.qubit<1>) {
      %qC_corr2 = "quantum.Z"(%qC_afterX_in) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
      "quantum.yield" (%qC_corr2) : (!quantum.qubit<1>) -> ()
    } else {
      "quantum.yield" (%qC_afterX_in) : (!quantum.qubit<1>) -> ()
    }

    // --- Verification ---
    // Measure Bob's qubit.
    %mC, %qC_meas = "quantum.measure_single"(%qC_corrected) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "quantum.deallocate"(%qA_meas) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%qB_meas) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%qC_meas) : (!quantum.qubit<1>) -> ()

    // Print the measurement of Bob's qubit.
    vector.print %mC : i1
    return
  }

  func.func @entry() {
    // The printed measurement should match the initial state prepared on qubit A.
    func.call @teleport() : () -> ()
    return
  }
}
