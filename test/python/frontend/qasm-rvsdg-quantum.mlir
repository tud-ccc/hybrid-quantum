// RUN: %PYTHON qasm-import -i %s -r | FileCheck %s

// CHECK: module {
// CHECK-LABEL:  qasm_main
// CHECK-DAG:    %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK-DAG:    %[[M0:.+]], %[[Q1:.+]] = "quantum.measure_single"(%[[Q0]]) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
// CHECK:    %[[false:.+]] = arith.constant false
// CHECK-DAG:    %[[cmpi:.+]] = arith.cmpi eq, %[[M0]], %[[false]] : i1
// CHECK-DAG:    %[[cond:.+]] = rvsdg.match(%[[cmpi]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
// CHECK-DAG:    %[[Q3:.+]] = rvsdg.gamma(%[[cond]] : <2>) (%[[Q1]]: !quantum.qubit<1>) : [
// CHECK-NEXT:      (%[[arg0:.+]]: !quantum.qubit<1>): {
// CHECK:        %[[Q4:.+]] = "quantum.X"(%[[arg0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK-NEXT:        rvsdg.yield (%[[Q4]]: !quantum.qubit<1>)
// CHECK-NEXT:      }, 
// CHECK-NEXT:      (%[[arg1:.+]]: !quantum.qubit<1>): {
// CHECK-NEXT:        rvsdg.yield (%[[arg1]]: !quantum.qubit<1>)
// CHECK-NEXT      }
// CHECK:    ] -> !quantum.qubit<1>
// CHECK-DAG:    %[[false2:.+]] = arith.constant false
// CHECK-DAG:    %[[cmpi2:.+]] = arith.cmpi eq, %[[M0]], %[[false2]] : i1
// CHECK-DAG:    %[[cond2:.+]] = rvsdg.match(%[[cmpi2]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
// CHECK-DAG:    %[[Q5:.+]] = rvsdg.gamma(%[[cond2]] : <2>) (%[[Q3]]: !quantum.qubit<1>) : [
// CHECK-NEXT:      (%[[arg2:.+]]: !quantum.qubit<1>): {
// CHECK:        %[[Q6:.+]] = "quantum.X"(%[[arg2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK-NEXT:        rvsdg.yield (%[[Q6]]: !quantum.qubit<1>)
// CHECK-NEXT:      }, 
// CHECK-NEXT:      (%[[arg3:.+]]: !quantum.qubit<1>): {
// CHECK-NEXT:        rvsdg.yield (%[[arg3]]: !quantum.qubit<1>)
// CHECK-NEXT:      }
// CHECK-NEXT:    ] -> !quantum.qubit<1>
// CHECK-DAG:    %[[M1:.+]], %[[Q7:.+]] = "quantum.measure_single"(%[[Q5]]) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
// CHECK-DAG:    %[[Q8:.+]] = "quantum.reset"(%[[Q7]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK-DAG:    "quantum.deallocate"(%[[Q8]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG:    %[[from_elements:.+]] = tensor.from_elements %[[M1]] : tensor<1xi1>
// CHECK-NEXT:    return %[[from_elements]] : tensor<1xi1>
// CHECK-NEXT:  }
// CHECK-NEXT: }

OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
if(c==0) x q[0];
if(c==0) x q[0];
measure q[0] -> c[0];
reset q[0];
