module {
  func.func @grover() {
    // Allocate 2 qubits and split
    %q_init = "quantum.alloc"() : () -> (!quantum.qubit<2>)
    %q0_0, %q1_0 = "quantum.split"(%q_init) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // Superposition
    %q0_1 = "quantum.H"(%q0_0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_1 = "quantum.H"(%q1_0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Oracle for |11⟩
    %q0_2, %q1_2 = "quantum.CNOT"(%q0_1, %q1_1) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q1_3 = "quantum.Z"(%q1_2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_3, %q1_4 = "quantum.CNOT"(%q0_2, %q1_3) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

    // Diffusion operator
    %q0_4 = "quantum.H"(%q0_3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_5 = "quantum.H"(%q1_4) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_5 = "quantum.X"(%q0_4) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_6 = "quantum.X"(%q1_5) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_6, %q1_7 = "quantum.CNOT"(%q0_5, %q1_6) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q1_8 = "quantum.Z"(%q1_7) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_7, %q1_9 = "quantum.CNOT"(%q0_6, %q1_8) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    %q0_8 = "quantum.X"(%q0_7) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_10 = "quantum.X"(%q1_9) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q0_9 = "quantum.H"(%q0_8) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q1_11 = "quantum.H"(%q1_10) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

    // Merge
    %q_final = "quantum.merge"(%q0_9, %q1_11) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)

    // Measure
    %m, %q_measured = "quantum.measure"(%q_final) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
    "quantum.deallocate"(%q_measured) : (!quantum.qubit<2>) -> ()

    // Extract bits
    %i0 = "index.constant" () {value = 0 : index} : () -> index
    %i1 = "index.constant" () {value = 1 : index} : () -> index
    %m0 = "tensor.extract"(%m, %i0) : (tensor<2xi1>, index) -> i1
    %m1 = "tensor.extract"(%m, %i1) : (tensor<2xi1>, index) -> i1

    vector.print %m0 : i1
    vector.print %m1 : i1

    return
  }

  func.func @entry() {
    func.call @grover() : () -> ()
    return
  }
}
