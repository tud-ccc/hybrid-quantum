// RUN: %PYTHON qasm-import -i %s | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK: "qir.gate"() <{function_type = (!qir.qubit, !qir.qubit, !qir.qubit) -> (), sym_name = "unmaj"}> ({
// CHECK-NEXT: ^bb0(%[[arg3:.+]]: {{.*}}, %[[arg4:.+]]: {{.*}}, %[[arg5:.+]]: {{.*}}):
// CHECK-DAG: "qir.CCX"(%[[arg3]], %[[arg4]], %[[arg5]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg5]], %[[arg3]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg3]], %[[arg4]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-NEXT: }) : () -> ()
//
// CHECK: "qir.gate"() <{function_type = (!qir.qubit, !qir.qubit, !qir.qubit) -> (), sym_name = "majority"}> ({
// CHECK-NEXT: ^bb0(%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}):
// CHECK-DAG: "qir.CNOT"(%[[arg2]], %[[arg1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg2]], %[[arg0]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CCX"(%[[arg0]], %[[arg1]], %[[arg2]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-NEXT: }) : () -> () 
//
// CHECK: "func.func"() <{function_type = () -> (), sym_name = "qasm_main", sym_visibility = "private"}> ({
// CHECK-DAG: %[[a0:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.X"(%[[a0]]) : (!qir.qubit) -> ()
// CHECK-DAG: %[[b0:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.X"(%[[b0]]) : (!qir.qubit) -> ()
// CHECK-DAG: %[[a1:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.call"(%[[a1]], %[[b0]], %[[a0]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> () 
// CHECK-DAG: "qir.CNOT"(%[[a1]], %[[a0]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: %[[b1:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.call"(%[[a1]], %[[b1]], %[[a0]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> () 
// CHECK-DAG: %[[ans0:.+]] = "qir.ralloc"() : () -> !qir.result
// CHECK-DAG: "qir.measure"(%[[b0]], %[[ans0]]) : (!qir.qubit, !qir.result) -> ()
// %[[bit:.+]] = "qir.read_measurement"(%[[ans0]]) : (!qir.result) -> (!tensor.tensor<1xi1>)
// CHECK:     return
// CHECK-NEXT:   }) : () -> () 
// CHECK-NEXT: }) : () -> () 
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
