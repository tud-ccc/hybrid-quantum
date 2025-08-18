// RUN: quantum-opt %s -inline -lift-qillr-to-quantum | FileCheck %s

module {
  // CHECK-NOT: "qillr.gate"
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit) -> (), sym_name = "syndrome"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit, %arg2: !qillr.qubit, %arg3: !qillr.qubit, %arg4: !qillr.qubit):
    "qillr.CNOT"(%arg0, %arg3) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%arg1, %arg3) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%arg1, %arg4) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%arg2, %arg4) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()

  // CHECK-LABEL: @qasm_main
  func.func @qasm_main() {
    %0 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.X"(%0) : (!qillr.qubit) -> ()
    %1 = "qillr.alloc"() : () -> !qillr.qubit
    %2 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.barrier"(%0, %1, %2) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    %3 = "qillr.alloc"() : () -> !qillr.qubit
    %4 = "qillr.alloc"() : () -> !qillr.qubit
    // CHECK-NOT: "qillr.call"
    "qillr.call"(%0, %1, %2, %3, %4) <{callee = @syndrome}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    %5 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%3, %5) : (!qillr.qubit, !qillr.result) -> ()
    %6 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %7 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%4, %7) : (!qillr.qubit, !qillr.result) -> ()
    %8 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %9 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %true = arith.constant true
    %10 = arith.cmpi eq, %9, %true : i1
    %11 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %false = arith.constant false
    %12 = arith.cmpi eq, %11, %false : i1
    %13 = arith.andi %10, %12 : i1
    scf.if %13 {
      "qillr.X"(%0) : (!qillr.qubit) -> ()
    }
    %14 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %false_0 = arith.constant false
    %15 = arith.cmpi eq, %14, %false_0 : i1
    %16 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %true_1 = arith.constant true
    %17 = arith.cmpi eq, %16, %true_1 : i1
    %18 = arith.andi %15, %17 : i1
    scf.if %18 {
      "qillr.X"(%2) : (!qillr.qubit) -> ()
    }
    %19 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %true_2 = arith.constant true
    %20 = arith.cmpi eq, %19, %true_2 : i1
    %21 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %true_3 = arith.constant true
    %22 = arith.cmpi eq, %21, %true_3 : i1
    %23 = arith.andi %20, %22 : i1
    scf.if %23 {
      "qillr.X"(%1) : (!qillr.qubit) -> ()
    }
    %24 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%0, %24) : (!qillr.qubit, !qillr.result) -> ()
    %25 = "qillr.read_measurement"(%24) : (!qillr.result) -> i1
    %26 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%1, %26) : (!qillr.qubit, !qillr.result) -> ()
    %27 = "qillr.read_measurement"(%26) : (!qillr.result) -> i1
    %28 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%2, %28) : (!qillr.qubit, !qillr.result) -> ()
    %29 = "qillr.read_measurement"(%28) : (!qillr.result) -> i1
    "qillr.reset"(%0) : (!qillr.qubit) -> ()
    "qillr.reset"(%1) : (!qillr.qubit) -> ()
    "qillr.reset"(%2) : (!qillr.qubit) -> ()
    "qillr.reset"(%3) : (!qillr.qubit) -> ()
    "qillr.reset"(%4) : (!qillr.qubit) -> ()
    return
  }
}
