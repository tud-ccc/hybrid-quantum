// RUN: python /net/media/scratch/quantum/hybrid-quantum/frontend/qasm/QASM2MLIR.py -i /net/media/scratch/quantum/hybrid-quantum/frontend/qasm/input.qasm | FileCheck %s

// CHECK-DAG: module {
// CHECK-DAG:   func.func @main() {

// CHECK-DAG:     %[[Q0:.*]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG:     %[[Q1:.*]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG:     %[[R0:.*]] = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG:     %[[R1:.*]] = "qir.ralloc" () : () -> (!qir.result)

// CHECK-DAG:     "qir.H" (%[[Q0]]) : (!qir.qubit) -> ()
// CHECK-DAG:     "qir.X" (%[[Q0]]) : (!qir.qubit) -> ()

// CHECK-DAG:     %[[THETA:.*]] = arith.constant 3.1415{{0*}} : f64
// CHECK-DAG:     "qir.Rx" (%[[Q0]], %[[THETA]]) : (!qir.qubit, f64) -> ()

// CHECK-DAG:     "qir.CNOT" (%[[Q0]], %[[Q1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG:     "qir.swap" (%[[Q0]], %[[Q1]]) : (!qir.qubit, !qir.qubit) -> ()

// CHECK-DAG:     "qir.measure" (%[[Q0]], %[[R0]]) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:     %[[M0:.*]] = "qir.read_measurement" (%[[R0]]) : (!qir.result) -> (!tensor.tensor<1xi1>)

// CHECK-DAG:     "qir.measure" (%[[Q1]], %[[R1]]) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:     %[[M1:.*]] = "qir.read_measurement" (%[[R1]]) : (!qir.result) -> (!tensor.tensor<1xi1>)

// CHECK-DAG:     "qir.reset" (%[[Q0]]) : (!qir.qubit) -> ()
// CHECK-DAG:     "qir.reset" (%[[Q1]]) : (!qir.qubit) -> ()

// CHECK-DAG:     return
// CHECK-DAG:   }
// CHECK-DAG: }
