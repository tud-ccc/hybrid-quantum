// RUN: %PYTHON qasm-import -i %s -r | FileCheck %s

//CHECK: module {
//CHECK:   qpu.module @qpu {
//CHECK:     "qpu.circuit"() <{function_type = () -> tensor<1xi1>, sym_name = "main"}> ({
//CHECK:       %[[QUBIT:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
//CHECK:       %[[CST:.+]] = arith.constant dense<false> : tensor<1xi1>
//CHECK:       %[[MEAS:.+]], %[[RES:.+]] = "quantum.measure"(%[[QUBIT]]) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
//CHECK:       %[[TENSOR:.+]] = "quantum.to_tensor"(%[[MEAS]]) : (!quantum.measurement<1>) -> tensor<1xi1>
//CHECK:       %[[INSERTED:.+]] = tensor.insert_slice %[[TENSOR]] into %[[CST]][0] [1] [1] : tensor<1xi1> into tensor<1xi1>
//CHECK:       %[[CST0:.+]] = arith.constant dense<false> : tensor<1xi1>
//CHECK:       %[[CMP0:.+]] = arith.cmpi eq, %[[INSERTED]], %[[CST0]] : tensor<1xi1>
//CHECK:       %[[IDX0:.+]] = arith.constant 0 : index
//CHECK:       %[[EXTRACT0:.+]] = tensor.extract %[[CMP0]][%[[IDX0]]] : tensor<1xi1>
//CHECK:       %[[MATCH0:.+]] = rvsdg.match(%[[EXTRACT0]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
//CHECK:       %[[GAMMA0:.+]] = rvsdg.gamma(%[[MATCH0]] : <2>) (%[[RES]]: !quantum.qubit<1>) : [
//CHECK:         (%[[ARG0:.+]]: !quantum.qubit<1>): {
//CHECK:           %[[X0:.+]] = "quantum.X"(%[[ARG0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
//CHECK:           rvsdg.yield (%[[X0]]: !quantum.qubit<1>)
//CHECK:         }, 
//CHECK:         (%[[ARG1:.+]]: !quantum.qubit<1>): {
//CHECK:           rvsdg.yield (%[[ARG1]]: !quantum.qubit<1>)
//CHECK:         }
//CHECK:       ] -> !quantum.qubit<1>
//CHECK:       %[[CST1:.+]] = arith.constant dense<false> : tensor<1xi1>
//CHECK:       %[[CMP1:.+]] = arith.cmpi eq, %[[INSERTED]], %[[CST1]] : tensor<1xi1>
//CHECK:       %[[IDX1:.+]] = arith.constant 0 : index
//CHECK:       %[[EXTRACT1:.+]] = tensor.extract %[[CMP1]][%[[IDX1]]] : tensor<1xi1>
//CHECK:       %[[MATCH1:.+]] = rvsdg.match(%[[EXTRACT1]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
//CHECK:       %[[GAMMA1:.+]] = rvsdg.gamma(%[[MATCH1]] : <2>) (%[[GAMMA0]]: !quantum.qubit<1>) : [
//CHECK:         (%[[ARG2:.+]]: !quantum.qubit<1>): {
//CHECK:           %[[X1:.+]] = "quantum.X"(%[[ARG2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
//CHECK:           rvsdg.yield (%[[X1]]: !quantum.qubit<1>)
//CHECK:         }, 
//CHECK:         (%[[ARG3:.+]]: !quantum.qubit<1>): {
//CHECK:           rvsdg.yield (%[[ARG3]]: !quantum.qubit<1>)
//CHECK:         }
//CHECK:       ] -> !quantum.qubit<1>
//CHECK:       %[[MEAS2:.+]], %[[RES2:.+]] = "quantum.measure"(%[[GAMMA1]]) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
//CHECK:       %[[TENSOR2:.+]] = "quantum.to_tensor"(%[[MEAS2]]) : (!quantum.measurement<1>) -> tensor<1xi1>
//CHECK:       %[[INSERTED2:.+]] = tensor.insert_slice %[[TENSOR2]] into %[[INSERTED]][0] [1] [1] : tensor<1xi1> into tensor<1xi1>
//CHECK:       %[[RESET:.+]] = "quantum.reset"(%[[RES2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
//CHECK:       "quantum.deallocate"(%[[RESET]]) : (!quantum.qubit<1>) -> ()
//CHECK:       "qpu.return"(%[[INSERTED2]]) : (tensor<1xi1>) -> ()
//CHECK:     }) : () -> ()
//CHECK:   }
//CHECK:   func.func public @qasm_main() -> tensor<1xi1> {
//CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<1xi1>
//CHECK:     %[[RES:.+]] = qpu.execute @qpu::@main ins () outs (%[[EMPTY]] : tensor<1xi1>)
//CHECK:     return %[[RES]] : tensor<1xi1>
//CHECK:   }
//CHECK: }


OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
if(c==0) x q[0];
if(c==0) x q[0];
measure q[0] -> c[0];
reset q[0];
