func.func @main() {
  // Loop 4 times—each iteration prepares one Bell state.
  affine.for %i = 0 to 4 {
    // Allocate two qubits for the Bell state.
    %q0 = "qir.alloc"() : () -> (!qir.qubit)
    %q1 = "qir.alloc"() : () -> (!qir.qubit)
    // Allocate two result registers for measurements.
    %r0 = "qir.ralloc"() : () -> (!qir.result)
    %r1 = "qir.ralloc"() : () -> (!qir.result)

    // Create constant indices for comparisons.
    %zero   = arith.constant 0 : index
    %one    = arith.constant 1 : index
    %two    = arith.constant 2 : index
    %three  = arith.constant 3 : index

    // For i == 0: Prepare |Φ+⟩:  H(q0); Cx(q0, q1)
    %cmp0 = arith.cmpi eq, %i, %zero : index
    scf.if %cmp0 -> () {
      "qir.H"(%q0) : (!qir.qubit) -> ()
      "qir.Cx"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
      scf.yield
    } else {
      // For i == 1: Prepare |Φ-⟩: H(q0); Z(q0); Cx(q0, q1)
      %cmp1 = arith.cmpi eq, %i, %one : index
      scf.if %cmp1 -> () {
        "qir.H"(%q0) : (!qir.qubit) -> ()
        "qir.Z"(%q0) : (!qir.qubit) -> ()
        "qir.Cx"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
        scf.yield
      } else {
        // For i == 2: Prepare |Ψ+⟩: H(q0); X(q1); Cx(q0, q1)
        %cmp2 = arith.cmpi eq, %i, %two : index
        scf.if %cmp2 -> () {
          "qir.H"(%q0) : (!qir.qubit) -> ()
          "qir.X"(%q1) : (!qir.qubit) -> ()
          "qir.Cx"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
          scf.yield
        } else {
          // For i == 3: Prepare |Ψ-⟩: H(q0); Z(q0); X(q1); Cx(q0, q1)
          %cmp3 = arith.cmpi eq, %i, %three : index
          scf.if %cmp3 -> () {
            "qir.H"(%q0) : (!qir.qubit) -> ()
            "qir.Z"(%q0) : (!qir.qubit) -> ()
            "qir.X"(%q1) : (!qir.qubit) -> ()
            "qir.Cx"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
            scf.yield
          }
        }
      }
    }

    // Measure both qubits.
    "qir.measure"(%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    "qir.measure"(%q1, %r1) : (!qir.qubit, !qir.result) -> ()
  }
  return
}
