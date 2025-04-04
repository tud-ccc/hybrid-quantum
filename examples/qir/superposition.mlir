//Example code to demonstrate quantum superposition. Should yield ~50% |0⟩ and ~50% |1⟩ over multiple runs
func.func @main()  -> i1 {
  // Allocate a qubit and a classical result register
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  %r0 = "qir.ralloc" () : () -> (!qir.result)

  // Apply Hadamard gate followed by CNOT to put q0, q1 into superposition: |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
  "qir.H" (%q0) : (!qir.qubit) -> ()
  "qir.CNOT" (%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()

  // Measure the qubit
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  %output = "qir.read_measurement"(%r0) : (!qir.result) -> tensor<1xi1>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %output[%c0] : tensor<1xi1>
  return %result : i1 
}
