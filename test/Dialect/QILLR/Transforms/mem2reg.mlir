// RUN: quantum-opt %s -debug --mlir-print-ir-after-all -mem2reg -split-input-file | FileCheck %s


  // CHECK-LABEL: func.func @simple_mem_to_reg(
  // CHECK-SAME: %[[Q:.+]]:{{.*}}, %[[T:.+]]:{{.*}}) -> !qillr.qubit {
  func.func @simple_mem_to_reg(%q :!qillr.qubit, %theta : f64) -> !qillr.qubit {
    // CHECK-NEXT: "qillr.Rz"(%[[Q]], %[[T]]) : (!qillr.qubit, f64) -> ()
    "qillr.Rz"(%q, %theta) : (!qillr.qubit, f64) -> ()
    // CHECK-DAG: return %[[Q]] : !qillr.qubit
    return %q : !qillr.qubit
  }

  // -----
