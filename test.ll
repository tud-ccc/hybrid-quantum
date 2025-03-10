module {
  func.func @main() -> i32 {
    %0 = "qir.alloc"() : () -> !qir.qubit
    "qir.H"(%0) : (!qir.qubit) -> ()
    %1 = "qir.ralloc"() : () -> !qir.result
    "qir.measure"(%0, %1) : (!qir.qubit, !qir.result) -> ()
    %2 = "qir.read_measurement"(%1) : (!qir.result) -> i1
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

