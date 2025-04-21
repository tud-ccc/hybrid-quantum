// RUN: python /net/media/scratch/quantum/hybrid-quantum/frontend/qasm/QASM2MLIR.py -i /net/media/scratch/quantum/hybrid-quantum/frontend/qasm/ifelse.qasm | FileCheck %s

// CHECK-DAG: module {
// CHECK-DAG:   func.func @main() {

// CHECK-DAG:   %[[Q0:[0-9]+]] = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG:   "qir.H" (%[[Q0]]) : (!qir.qubit) -> ()
// CHECK-DAG:   %[[R0:[0-9]+]] = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG:   "qir.measure" (%[[Q0]], %[[R0]]) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:   %[[M0:[0-9]+]] = "qir.read_measurement" (%[[R0]]) : (!qir.result) -> (!tensor.tensor<1xi1>)
// CHECK-DAG:   %[[C1:[0-9]+]] = arith.constant 1.000000 : i1
// CHECK-DAG:   %[[P:[0-9]+]] = "arith.cmpi" (%[[R0]], %[[C1]]) : (!qir.result, i1) -> (i1)
// CHECK-DAG:   scf.if %[[P]] {
// CHECK-DAG:     "qir.X" (%[[Q0]]) : (!qir.qubit) -> ()
// CHECK-DAG:   }
// CHECK-DAG:   return
// CHECK-DAG:   }
// CHECK-DAG: }