// RUN: %PYTHON qasm-import -i %s | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK: "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> (), sym_name = "unmaj"}> ({
// CHECK-NEXT: ^bb0(%[[arg3:.+]]: {{.*}}, %[[arg4:.+]]: {{.*}}, %[[arg5:.+]]: {{.*}}):
// CHECK-DAG: "qillr.CCX"(%[[arg3]], %[[arg4]], %[[arg5]]) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CNOT"(%[[arg5]], %[[arg3]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CNOT"(%[[arg3]], %[[arg4]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK: "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> (), sym_name = "majority"}> ({
// CHECK-NEXT: ^bb0(%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}):
// CHECK-DAG: "qillr.CNOT"(%[[arg2]], %[[arg1]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CNOT"(%[[arg2]], %[[arg0]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CCX"(%[[arg0]], %[[arg1]], %[[arg2]]) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK: "func.func"() <{function_type = () -> (), sym_name = "qasm_main", sym_visibility = "private"}> ({
// CHECK-DAG: %[[a0:.+]] = "qillr.alloc"() : () -> !qillr.qubit
// CHECK-DAG: "qillr.X"(%[[a0]]) : (!qillr.qubit) -> ()
// CHECK-DAG: %[[b0:.+]] = "qillr.alloc"() : () -> !qillr.qubit
// CHECK-DAG: "qillr.X"(%[[b0]]) : (!qillr.qubit) -> ()
// CHECK-DAG: %[[a1:.+]] = "qillr.alloc"() : () -> !qillr.qubit
// CHECK-DAG: "qillr.call"(%[[a1]], %[[b0]], %[[a0]]) <{callee = @majority}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CNOT"(%[[a1]], %[[a0]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: %[[b1:.+]] = "qillr.alloc"() : () -> !qillr.qubit
// CHECK-DAG: "qillr.call"(%[[a1]], %[[b1]], %[[a0]]) <{callee = @unmaj}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()

// CHECK-DAG: "qillr.H"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Y"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Z"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.S"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Sdg"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.T"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Tdg"(%[[a1]]) : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Rx"(%[[a1]], %{{.+}}) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.Ry"(%[[a1]], %{{.+}}) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.Rz"(%[[a1]], %{{.+}}) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.U1"(%[[a1]], %{{.+}}) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.U2"(%[[a1]], %{{.+}}, %{{.+}}) : (!qillr.qubit, f64, f64) -> ()
// CHECK-DAG: "qillr.U3"(%[[a1]], %{{.+}}, %{{.+}}, %{{.+}}) : (!qillr.qubit, f64, f64, f64) -> ()
// CHECK-DAG: "qillr.Cz"(%[[a1]], %[[b1]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.swap"(%[[a1]], %[[b1]]) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CRz"(%[[a1]], %[[b1]], %{{.+}}) : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.CRy"(%[[a1]], %[[b1]], %{{.+}}) : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.barrier"(%[[a1]]) : (!qillr.qubit) -> ()

// CHECK-DAG: %[[ans0:.+]] = "qillr.ralloc"() : () -> !qillr.result
// CHECK-DAG: "qillr.measure"(%[[a1]], %[[ans0]]) : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: %[[bit:.+]] = "qillr.read_measurement"(%[[ans0]]) : (!qillr.result) -> i1
// CHECK-DAG: "qillr.reset"(%[[a1]]) : (!qillr.qubit) -> ()
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
