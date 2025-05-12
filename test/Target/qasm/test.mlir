// RUN: quantum-translate --mlir-to-openqasm %s | FileCheck %s

module {
  // Allocate two qubits
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
}

// CHECK-DAG: OPENQASM 2.0;
// CHECK-DAG: include "qelib1.inc";
// CHECK-DAG: qreg q0[1];
// CHECK-DAG: qreg q1[1];
