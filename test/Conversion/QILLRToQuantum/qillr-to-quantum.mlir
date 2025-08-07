// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file | FileCheck %s

// CHECK-LABEL: test_H
func.func @test_H() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q3:.+]] = "quantum.H"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q3]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.H" (%q) : (!qillr.qubit) -> ()
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_rot
func.func @test_rot() {
// CHECK-DAG: %[[cst1:.+]] = arith.constant
%const1 = arith.constant 0.34 : f64
// CHECK-DAG: %[[cst2:.+]] = arith.constant
%const2 = arith.constant 0.78 : f64
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q3:.+]] = "quantum.Rx"(%[[Q2]], %[[cst1]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q3]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.Rx" (%q, %const1) : (!qillr.qubit, f64) -> ()
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q5:.+]] = "quantum.Ry"(%[[Q4]], %[[cst2]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q5]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.Ry" (%q, %const2) : (!qillr.qubit, f64) -> ()
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q7:.+]] = "quantum.Rz"(%[[Q6]], %[[cst1]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q7]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.Rz" (%q, %const1) : (!qillr.qubit, f64) -> ()
// CHECK: %[[Q8:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q8]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_swap
func.func @test_swap() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q2]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q3:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q4:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q5:.+]], %[[Q6:.+]] = "quantum.SWAP"(%[[Q3]], %[[Q4]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
// CHECK-DAG: "qqt.store"(%[[Q5]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q6]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.swap"(%q1, %q2) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK: %[[Q7:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q7]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q8:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q8]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_cz
func.func @test_cz() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q2]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q3:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q4:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q5:.+]], %[[Q6:.+]] = "quantum.CZ"(%[[Q3]], %[[Q4]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
// CHECK-DAG: "qqt.store"(%[[Q5]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q6]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.Cz"(%q1, %q2) : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK: %[[Q7:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q7]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q8:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q8]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_ccx
func.func @test_ccx() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q2]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref3:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q3:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q3]], %[[Ref3]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q3 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q5:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q6:.+]] = "qqt.load"(%[[Ref3]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q7:.+]], %[[Q8:.+]], %[[Q9:.+]] = "quantum.CCX"(%[[Q4]], %[[Q5]], %[[Q6]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
// CHECK-DAG: "qqt.store"(%[[Q7]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q8]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q9]], %[[Ref3]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.CCX"(%q1, %q2, %q3) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK: %[[Q10:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q10]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q11:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q11]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
// CHECK: %[[Q12:.+]] = "qqt.load"(%[[Ref3]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q12]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref3]]) : (!qqt.ref) -> () 
"qillr.reset" (%q3) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_barrier
func.func @test_barrier() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q2]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref3:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q3:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q3]], %[[Ref3]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q3 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q5:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q6:.+]] = "qqt.load"(%[[Ref3]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q7:.+]]:3 = "quantum.barrier"(%[[Q4]], %[[Q5]], %[[Q6]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
// CHECK-DAG: "qqt.store"(%[[Q7]]#{{.*}}, %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q7]]#{{.*}}, %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[Q7]]#{{.*}}, %[[Ref3]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.barrier"(%q1, %q2, %q3) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK: %[[Q8:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q8]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q9:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q9]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
// CHECK: %[[Q10:.+]] = "qqt.load"(%[[Ref3]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q10]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref3]]) : (!qqt.ref) -> () 
"qillr.reset" (%q3) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: test_measure
func.func @test_measure() -> tensor<1xi1> {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NOT: "qillr.ralloc"
%r = "qillr.ralloc" () : () -> (!qillr.result)
// CHECK-DAG: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[M:.+]], %[[Q3:.+]] = "quantum.measure_single"(%[[Q2]]) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
// CHECK-DAG: "qqt.store"(%[[Q3]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.measure" (%q1, %r) : (!qillr.qubit, !qillr.result) -> ()
// CHECK-NOT: "qillr.read_measurement"
%m = "qillr.read_measurement" (%r) : (!qillr.result) -> i1
// CHECK-DAG: %[[MT:.+]] = tensor.from_elements %[[M]] : tensor<1xi1>
%mt = tensor.from_elements %m : tensor<1xi1>
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return %mt : tensor<1xi1>
}

// -----
