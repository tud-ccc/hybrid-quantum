// RUN: quantum-opt %s --convert-quantum-to-qillr -split-input-file | FileCheck %s
// RUN: quantum-opt %s --convert-quantum-to-qillr --canonicalize -split-input-file | FileCheck %s --check-prefix=CANON

// CHECK-LABEL: func.func @scf_if(
// CANON-LABEL: func.func @scf_if(
func.func @scf_if(%b : i1) {
    // CHECK: %[[Q:.*]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
    %q = "quantum.alloc"() : () -> !quantum.qubit<1>

    // CHECK: %[[QOUT:.*]] = scf.if {{.*}} -> (!qillr.qubit) {
    // CANON: scf.if {{.*}} {
    %q2 = scf.if %b -> (!quantum.qubit<1>) {
        // CHECK: "qillr.X"(%[[Q]]) <{index = [0]}> : (!qillr.qubit) -> ()
        %qX = "quantum.X"(%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>

        // CHECK: scf.yield %[[Q]] : !qillr.qubit
        // CANON-NOT: scf.yield
        scf.yield %qX : !quantum.qubit<1>
    } else {
        // CHECK: "qillr.Z"(%[[Q]]) <{index = [0]}> : (!qillr.qubit) -> ()
        %qZ = "quantum.Z"(%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>

        // CHECK: scf.yield %[[Q]] : !qillr.qubit
        // CANON-NOT: scf.yield
        scf.yield %qZ : !quantum.qubit<1>
    }

    // CANON: return
    return
}

// -----

// CHECK-LABEL: func.func @scf_if_qillr(
// CANON-LABEL: func.func @scf_if_qillr(
func.func @scf_if_qillr(%b : i1) {
    // CHECK: %[[Q:.*]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
    // CANON: %[[Q:.*]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
    %q = "quantum.alloc"() : () -> !quantum.qubit<1>

    // CHECK: %[[QOUT:.*]] = scf.if {{.*}} -> (!qillr.qubit) {
    // CANON: scf.if {{.*}} {
    %q2 = scf.if %b -> (!quantum.qubit<1>) {
        // CHECK: "qillr.X"(%[[Q]]) <{index = [0]}> : (!qillr.qubit) -> ()
        %qX = "quantum.X"(%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>

        // CHECK: scf.yield %[[Q]] : !qillr.qubit
        // CANON-NOT: scf.yield
        scf.yield %qX : !quantum.qubit<1>
    } else {
        // CHECK: scf.yield %[[Q]] : !qillr.qubit
        // CANON-NOT: scf.yield
        scf.yield %q : !quantum.qubit<1>
    }

    // CHECK: "qillr.H"(%[[QOUT]]) <{index = [0]}> : (!qillr.qubit) -> ()
    // CANON: "qillr.H"(%[[Q]]) <{index = [0]}> : (!qillr.qubit) -> ()
    %q3 = "quantum.H"(%q2) : (!quantum.qubit<1>) -> !quantum.qubit<1>

    // CHECK: "qillr.deallocate"(%[[QOUT]]) <{index = [0]}> : (!qillr.qubit) -> ()
    // CANON: "qillr.deallocate"(%[[Q]]) <{index = [0]}> : (!qillr.qubit) -> ()
    "quantum.deallocate"(%q3) : (!quantum.qubit<1>) -> ()

    // CANON: return
    return
}
