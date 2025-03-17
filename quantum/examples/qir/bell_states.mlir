module {
  func.func @main() -> (i1, i1, i1, i1) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %step = arith.constant 1 : index
    %init = arith.constant 0 : i1

    // scf.for loop iterating 4 times to prepare 4 Bell states
    %m0, %m1, %m2, %m3 = scf.for %i = %c0 to %c4 step %step iter_args(%r0 = %init, %r1 = %init, %r2 = %init, %r3 = %init) -> (i1, i1, i1, i1) {
      // Allocate two qubits for Bell state preparation
      %q0 = "qir.alloc"() : () -> (!qir.qubit)
      %q1 = "qir.alloc"() : () -> (!qir.qubit)

      // Allocate result registers
      %res0 = "qir.ralloc"() : () -> (!qir.result)
      %res1 = "qir.ralloc"() : () -> (!qir.result)

      // Prepare Bell states based on loop index
      %zero  = arith.constant 0 : index
      %one   = arith.constant 1 : index
      %two   = arith.constant 2 : index
      %three = arith.constant 3 : index

      %cmp0 = arith.cmpi eq, %i, %zero : index
      scf.if %cmp0 {
        "qir.H"(%q0) : (!qir.qubit) -> ()
        "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
      }

      %cmp1 = arith.cmpi eq, %i, %one : index
      scf.if %cmp1 {
        "qir.H"(%q0) : (!qir.qubit) -> ()
        "qir.Z"(%q0) : (!qir.qubit) -> ()
        "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
      }

      %cmp2 = arith.cmpi eq, %i, %two : index
      scf.if %cmp2 {
        "qir.H"(%q0) : (!qir.qubit) -> ()
        "qir.X"(%q1) : (!qir.qubit) -> ()
        "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
      }

      %cmp3 = arith.cmpi eq, %i, %three : index
      scf.if %cmp3 {
        "qir.H"(%q0) : (!qir.qubit) -> ()
        "qir.Z"(%q0) : (!qir.qubit) -> ()
        "qir.X"(%q1) : (!qir.qubit) -> ()
        "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
      }

      // Measure the qubits
      "qir.measure"(%q0, %res0) : (!qir.qubit, !qir.result) -> ()
      "qir.measure"(%q1, %res1) : (!qir.qubit, !qir.result) -> ()

      // Convert measurement results to classical bits
      %m_out0 = "qir.read_measurement"(%res0) : (!qir.result) -> (i1)
      %m_out1 = "qir.read_measurement"(%res1) : (!qir.result) -> (i1)

      // Yield results to accumulate them across iterations
      scf.yield %m_out0, %m_out1, %r2, %r3 : i1, i1, i1, i1
    }

    // Return final measurement results after loop execution
    return %m0, %m1, %m2, %m3 : i1, i1, i1, i1
  }
}
