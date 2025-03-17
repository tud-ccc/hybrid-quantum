//Example code to demonstrate quantum superposition. Should yield ~50% |0⟩ and ~50% |1⟩ over multiple runs

func.func @main() {
  // Allocate a qubit and a classical result register
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  %q3 = "qir.alloc" () : () -> (!qir.qubit)
  %q2 = "qir.alloc" () : () -> (!qir.qubit)

  %r0 = "qir.ralloc" () : () -> (!qir.result)

  // Apply Hadamard gate to put q0 into superposition: |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
  "qir.H" (%q0) : (!qir.qubit) -> ()

  // Measure the qubit

  return
}
