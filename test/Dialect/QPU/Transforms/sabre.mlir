// RUN: quantum-opt %s --sabre-swap | FileCheck %s

module {
  qpu.module @test [#qpu.target<qubits = 3 : i64, coupling_graph = [[0, 1], [1, 2]]>] {
    "qpu.circuit"() <{function_type = () -> (), sym_name = "test_circuit"}> ({
      ^bb0():
        // 1) allocate three qubits
        %q0 = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q1 = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q2 = "quantum.alloc"() : () -> !quantum.qubit<1>

        // 2) adjacent CNOT on (%q0, %q1)
        %c0, %c1 = "quantum.CNOT"(%q0, %q1)
          : (!quantum.qubit<1>, !quantum.qubit<1>)
          -> (!quantum.qubit<1>, !quantum.qubit<1>)

        // 3) non-adjacent CNOT on (%c0, %q2)
        %c2, %c3 = "quantum.CNOT"(%c0, %q2)
          : (!quantum.qubit<1>, !quantum.qubit<1>)
          -> (!quantum.qubit<1>, !quantum.qubit<1>)

        "qpu.return"() : () -> ()
    }) : () -> ()
  }
}

// CHECK-DAG: %[[Q0:.*]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK-DAG: %[[Q1:.*]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK-DAG: %[[Q2:.*]] = "quantum.alloc"() : () -> !quantum.qubit<1>

// CHECK-DAG: %[[CTRL:.*]], %[[TGT:.*]] = "quantum.CNOT"(%[[Q0]], %[[Q1]])
// CHECK-DAG: %[[S0:.*]], %[[S1:.*]] = "quantum.SWAP"(%[[CTRL]], %[[TGT]])
// CHECK-DAG: %[[C2:.*]], %[[C3:.*]] = "quantum.CNOT"(%[[S1]], %[[Q2]])
