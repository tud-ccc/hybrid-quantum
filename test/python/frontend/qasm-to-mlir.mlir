// RUN: %PYTHON qasm-import -i %s | FileCheck %s

// CHECK: module {
// CHECK:   func.func @main()

// Allocate and init a[0], b[0]
// CHECK-DAG: %[[Q0:[0-9]+]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: %[[Q1:[0-9]+]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: %[[Q2:[0-9]+]] = "qir.alloc" () : () -> (!qir.qubit)

// CHECK-DAG: "qir.X" (%[[Q0]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.X" (%[[Q1]]) : (!qir.qubit) -> ()

// majority a[1],b[0],a[0]:
// CHECK-DAG: "qir.CNOT" (%[[Q0]], %[[Q1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT" (%[[Q0]], %[[Q2]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CCX" (%[[Q2]], %[[Q1]], %[[Q0]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()

// the standalone cx a[1],a[0]
// CHECK-DAG: "qir.CNOT" (%[[Q2]], %[[Q0]]) : (!qir.qubit, !qir.qubit) -> ()

// unmaj a[1],b[1],a[0]:
// ccx a[1],b[1],a[0]; cx a[0],a[1]; cx a[1],b[1]
// CHECK-DAG: %[[Q3:[0-9]+]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: "qir.CCX" (%[[Q2]], %[[Q3]], %[[Q0]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT" (%[[Q0]], %[[Q2]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT" (%[[Q2]], %[[Q3]]) : (!qir.qubit, !qir.qubit) -> ()

// Measure b[0] -> ans[0]
// CHECK-DAG: %[[R0:[0-9]+]] = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG: "qir.measure" (%[[Q1]], %[[R0]]) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG: %[[M0:[0-9]+]] = "qir.read_measurement" (%[[R0]]) : (!qir.result) -> (!tensor.tensor<1xi1>)

// CHECK:     return
// CHECK:   }
// CHECK: }

OPENQASM 2.0;
include "qelib1.inc";
gate majority a,b,c 
{ 
  cx c,b; 
  cx c,a; 
  ccx a,b,c; 
}

gate unmaj a,b,c 
{ 
  ccx a,b,c; 
  cx c,a; 
  cx a,b; 
}

qreg a[2];
qreg b[2];
creg ans[2];

x a[0];    // a = 0001
x b[0];    // b = 1111

majority a[1],b[0],a[0];
cx a[1],a[0];
unmaj a[1],b[1],a[0];

measure b[0] -> ans[0];
