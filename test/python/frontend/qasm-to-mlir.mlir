// RUN: %PYTHON qasm-import -i %s | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK: "qir.gate"() <{function_type = (!qir.qubit, !qir.qubit, !qir.qubit) -> (), sym_name = "unmaj"}> ({
// CHECK-NEXT: ^bb0(%[[arg3:.+]]: {{.*}}, %[[arg4:.+]]: {{.*}}, %[[arg5:.+]]: {{.*}}):
// CHECK-DAG: "qir.CCX"(%[[arg3]], %[[arg4]], %[[arg5]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg5]], %[[arg3]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg3]], %[[arg4]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK: "qir.gate"() <{function_type = (!qir.qubit, !qir.qubit, !qir.qubit) -> (), sym_name = "majority"}> ({
// CHECK-NEXT: ^bb0(%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}):
// CHECK-DAG: "qir.CNOT"(%[[arg2]], %[[arg1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[arg2]], %[[arg0]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CCX"(%[[arg0]], %[[arg1]], %[[arg2]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK: "func.func"() <{function_type = () -> (), sym_name = "qasm_main", sym_visibility = "private"}> ({
// CHECK-DAG: %[[a0:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.X"(%[[a0]]) : (!qir.qubit) -> ()
// CHECK-DAG: %[[b0:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.X"(%[[b0]]) : (!qir.qubit) -> ()
// CHECK-DAG: %[[a1:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.call"(%[[a1]], %[[b0]], %[[a0]]) <{callee = @majority}> : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[a1]], %[[a0]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: %[[b1:.+]] = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG: "qir.call"(%[[a1]], %[[b1]], %[[a0]]) <{callee = @unmaj}> : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()

// CHECK-DAG: "qir.H"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Y"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Z"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.S"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Sdg"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.T"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Tdg"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Rx"(%[[a1]], %{{.+}}) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.Ry"(%[[a1]], %{{.+}}) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.Rz"(%[[a1]], %{{.+}}) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.U1"(%[[a1]], %{{.+}}) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.U2"(%[[a1]], %{{.+}}, %{{.+}}) : (!qir.qubit, f64, f64) -> ()
// CHECK-DAG: "qir.U3"(%[[a1]], %{{.+}}, %{{.+}}, %{{.+}}) : (!qir.qubit, f64, f64, f64) -> ()
// CHECK-DAG: "qir.Cz"(%[[a1]], %[[b1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.swap"(%[[a1]], %[[b1]]) : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CRz"(%[[a1]], %[[b1]], %{{.+}}) : (!qir.qubit, !qir.qubit, f64) -> ()
// CHECK-DAG: "qir.CRy"(%[[a1]], %[[b1]], %{{.+}}) : (!qir.qubit, !qir.qubit, f64) -> ()
// CHECK-DAG: "qir.barrier"(%[[a1]]) : (!qir.qubit) -> ()

// CHECK-DAG: %[[ans0:.+]] = "qir.ralloc"() : () -> !qir.result
// CHECK-DAG: "qir.measure"(%[[a1]], %[[ans0]]) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG: %[[bit:.+]] = "qir.read_measurement"(%[[ans0]]) : (!qir.result) -> tensor<1xi1>
// CHECK-DAG: "qir.reset"(%[[a1]]) : (!qir.qubit) -> ()
// CHECK: return
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

OPENQASM 2.0;
include "qelib1.inc";
//test gates definition
gate majority a,b,c {
  cx c,b;
  cx c,a;
  ccx a,b,c;
}

gate unmaj a,b,c {
  ccx a,b,c;
  cx c,a;
  cx a,b;
}

qreg a[2];
qreg b[2];
creg ans[2];

x a[0];
x b[0];

//test gate calls
majority a[1], b[0], a[0];
cx a[1], a[0];
unmaj a[1], b[1], a[0];

// Apply standard gates to a[1] and b[1]
h a[1];
y a[1];
z a[1];
s a[1];
sdg a[1];
t a[1];
tdg a[1];
rx(pi/4) a[1];
ry(pi/8) a[1];
rz(pi/6) a[1];
u1(pi/3) a[1];
u2(pi/2, pi/4) a[1];
u3(pi/2, pi/3, pi/4) a[1];
cz a[1], b[1];
swap a[1], b[1];
crz(pi/2) a[1], b[1];
cry(pi/4) a[1], b[1];
barrier a[1];

measure a[1] -> ans[0];
reset a[1];
