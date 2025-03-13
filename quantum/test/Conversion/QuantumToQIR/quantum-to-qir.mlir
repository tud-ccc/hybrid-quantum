// RUN: quantum-opt %s --convert-quantum-to-qir --mlir-print-ir-after-all | FileCheck %s
//--debug-only=dialect-conversion

module {

    // CHECK-LABEL: func.func @return_single_qubit(
    // CHECK: ) -> !qir.qubit {
    func.func @return_single_qubit() -> (!quantum.qubit<1>) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-NEXT: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-NEXT: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> i1
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-NEXT: return %[[Q]]
        func.return %q_m : !quantum.qubit<1>
    }

    // CHECK-LABEL: func.func @return_single_measurement_result(
    // CHECK: ) -> i1 {
    func.func @return_single_measurement_result() -> (i1) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-NEXT: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-NEXT: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> i1
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-NEXT: return %[[M]]
        func.return %m : i1
    }

    // CHECK-LABEL: func.func @convertHOp(
    func.func @convertHOp() -> (i1) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: "qir.H"(%[[Q]]) : (!qir.qubit) -> ()
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-NEXT: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-NEXT: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-NEXT: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> i1
        %m, %q_m = "quantum.measure" (%q1) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-NEXT: return %[[M]]
        return %m : i1
    }

    // CHECK-LABEL: func.func @convertHOp2(
    func.func @convertHOp2() -> (!quantum.qubit<1>) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: "qir.H"(%[[Q]]) : (!qir.qubit) -> ()
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-NEXT: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-NEXT: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-NEXT: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> i1
        %m, %q_m = "quantum.measure" (%q1) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-NEXT: return %[[Q]]
        return %q_m : !quantum.qubit<1>
    }

    // CHECK-LABEL: func.func @convertSwap(

    func.func @convertSwap() -> () {
      // CHECK-NEXT: %[[Q1:[0]+]] = "qir.alloc"() : () -> !qir.qubit
      // CHECK-NEXT: %[[Q2:[1]+]] = "qir.alloc"() : () -> !qir.qubit
      %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
      %q2 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
      // CHECK-NEXT: "qir.Swap"(%[[Q1]], %[[Q2]]) : (!qir.qubit, !qir.qubit) -> ()
      %q1_out, %q2_out = "quantum.SWAP"(%q1, %q2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
      return
    }
}
