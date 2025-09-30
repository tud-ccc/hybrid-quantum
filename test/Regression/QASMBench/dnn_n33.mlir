// RUN: quantum-opt %s -inline -replace-repeated-reads -lift-qillr-to-quantum -hoist-load-store -eliminate-load-store  | FileCheck %s
//
//--debug --mlir-print-ir-after-all
module {
  // CHECK-NOT: "qillr.gate"
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959794112"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 1.5327306000000001 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959794016"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 6.1439234999999996 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793920"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 4.3981827999999998 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793824"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 3.7752618999999998 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793728"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 0.73078809 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793632"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 4.2819414 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793536"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 2.8259273999999999 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793440"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 3.9713219 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793344"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 4.6165576000000001 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793248"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 5.2693953999999996 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793152"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 5.7884126 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959793056"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 4.4686656999999999 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959792960"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 3.7841355000000001 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy_140221959791232"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 5.9976247000000003 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  "qillr.gate"() <{function_type = (!qillr.qubit, !qillr.qubit) -> (), sym_name = "ryy"}> ({
  ^bb0(%arg0: !qillr.qubit, %arg1: !qillr.qubit):
    %cst = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_0) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_1 = arith.constant 4.0860617000000001 : f64
    "qillr.Rz"(%arg1, %cst_1) : (!qillr.qubit, f64) -> ()
    "qillr.CNOT"(%arg0, %arg1) : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_2 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg0, %cst_2) : (!qillr.qubit, f64) -> ()
    %cst_3 = arith.constant -1.5707963267948966 : f64
    "qillr.Rx"(%arg1, %cst_3) : (!qillr.qubit, f64) -> ()
    "qillr.return"() : () -> ()
  }) : () -> ()
  // CHECK-LABEL: @qasm_main
  func.func public @qasm_main() {
    %0 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.H"(%0) : (!qillr.qubit) -> ()
    %1 = "qillr.alloc"() : () -> !qillr.qubit
    %cst = arith.constant 2.5715826000000002 : f64
    "qillr.Ry"(%1, %cst) : (!qillr.qubit, f64) -> ()
    %cst_0 = arith.constant 2.6217923000000001 : f64
    "qillr.Rz"(%1, %cst_0) : (!qillr.qubit, f64) -> ()
    %2 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_1 = arith.constant 2.4644602 : f64
    "qillr.Ry"(%2, %cst_1) : (!qillr.qubit, f64) -> ()
    %cst_2 = arith.constant 5.5738716000000004 : f64
    "qillr.Rz"(%2, %cst_2) : (!qillr.qubit, f64) -> ()
    %3 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_3 = arith.constant 3.5775204999999999 : f64
    "qillr.Ry"(%3, %cst_3) : (!qillr.qubit, f64) -> ()
    %cst_4 = arith.constant 4.7796126000000001 : f64
    "qillr.Rz"(%3, %cst_4) : (!qillr.qubit, f64) -> ()
    %4 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_5 = arith.constant 1.5555192 : f64
    "qillr.Ry"(%4, %cst_5) : (!qillr.qubit, f64) -> ()
    %cst_6 = arith.constant 4.2675644999999998 : f64
    "qillr.Rz"(%4, %cst_6) : (!qillr.qubit, f64) -> ()
    %5 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_7 = arith.constant 1.2023211 : f64
    "qillr.Ry"(%5, %cst_7) : (!qillr.qubit, f64) -> ()
    %cst_8 = arith.constant 6.2501756000000004 : f64
    "qillr.Rz"(%5, %cst_8) : (!qillr.qubit, f64) -> ()
    %6 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_9 = arith.constant 0.1623173 : f64
    "qillr.Ry"(%6, %cst_9) : (!qillr.qubit, f64) -> ()
    %cst_10 = arith.constant 4.9516061000000002 : f64
    "qillr.Rz"(%6, %cst_10) : (!qillr.qubit, f64) -> ()
    %7 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_11 = arith.constant 3.4428318 : f64
    "qillr.Ry"(%7, %cst_11) : (!qillr.qubit, f64) -> ()
    %cst_12 = arith.constant 0.63921594000000004 : f64
    "qillr.Rz"(%7, %cst_12) : (!qillr.qubit, f64) -> ()
    %8 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_13 = arith.constant 2.464220e+00 : f64
    "qillr.Ry"(%8, %cst_13) : (!qillr.qubit, f64) -> ()
    %cst_14 = arith.constant 1.7315225000000001 : f64
    "qillr.Rz"(%8, %cst_14) : (!qillr.qubit, f64) -> ()
    %9 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_15 = arith.constant 5.2351172999999998 : f64
    "qillr.Ry"(%9, %cst_15) : (!qillr.qubit, f64) -> ()
    %cst_16 = arith.constant 3.2685827999999999 : f64
    "qillr.Rz"(%9, %cst_16) : (!qillr.qubit, f64) -> ()
    %10 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_17 = arith.constant 2.4811442000000001 : f64
    "qillr.Ry"(%10, %cst_17) : (!qillr.qubit, f64) -> ()
    %cst_18 = arith.constant 4.5889673999999996 : f64
    "qillr.Rz"(%10, %cst_18) : (!qillr.qubit, f64) -> ()
    %11 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_19 = arith.constant 0.47273595000000002 : f64
    "qillr.Ry"(%11, %cst_19) : (!qillr.qubit, f64) -> ()
    %cst_20 = arith.constant 1.6226202000000001 : f64
    "qillr.Rz"(%11, %cst_20) : (!qillr.qubit, f64) -> ()
    %12 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_21 = arith.constant 5.0391697000000004 : f64
    "qillr.Ry"(%12, %cst_21) : (!qillr.qubit, f64) -> ()
    %cst_22 = arith.constant 3.6154039 : f64
    "qillr.Rz"(%12, %cst_22) : (!qillr.qubit, f64) -> ()
    %13 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_23 = arith.constant 4.4137940999999996 : f64
    "qillr.Ry"(%13, %cst_23) : (!qillr.qubit, f64) -> ()
    %cst_24 = arith.constant 5.6435805999999999 : f64
    "qillr.Rz"(%13, %cst_24) : (!qillr.qubit, f64) -> ()
    %14 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_25 = arith.constant 2.4002674000000002 : f64
    "qillr.Ry"(%14, %cst_25) : (!qillr.qubit, f64) -> ()
    %cst_26 = arith.constant 5.1529404000000003 : f64
    "qillr.Rz"(%14, %cst_26) : (!qillr.qubit, f64) -> ()
    %15 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_27 = arith.constant 5.1921691000000001 : f64
    "qillr.Ry"(%15, %cst_27) : (!qillr.qubit, f64) -> ()
    %cst_28 = arith.constant 5.8935817999999998 : f64
    "qillr.Rz"(%15, %cst_28) : (!qillr.qubit, f64) -> ()
    %16 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_29 = arith.constant 2.5612355999999998 : f64
    "qillr.Ry"(%16, %cst_29) : (!qillr.qubit, f64) -> ()
    %cst_30 = arith.constant 5.1255661999999997 : f64
    "qillr.Rz"(%16, %cst_30) : (!qillr.qubit, f64) -> ()
    "qillr.call"(%1, %2) <{callee = @ryy}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%2, %3) <{callee = @ryy_140221959791232}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%3, %4) <{callee = @ryy_140221959792960}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%4, %5) <{callee = @ryy_140221959793056}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%5, %6) <{callee = @ryy_140221959793152}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%6, %7) <{callee = @ryy_140221959793248}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%7, %8) <{callee = @ryy_140221959793344}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%8, %9) <{callee = @ryy_140221959793440}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%9, %10) <{callee = @ryy_140221959793536}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%10, %11) <{callee = @ryy_140221959793632}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%11, %12) <{callee = @ryy_140221959793728}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%12, %13) <{callee = @ryy_140221959793824}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%13, %14) <{callee = @ryy_140221959793920}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%14, %15) <{callee = @ryy_140221959794016}> : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.call"(%15, %16) <{callee = @ryy_140221959794112}> : (!qillr.qubit, !qillr.qubit) -> ()
    %cst_31 = arith.constant 4.7136841 : f64
    "qillr.CRy"(%1, %2, %cst_31) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_32 = arith.constant 5.7765275000000003 : f64
    "qillr.CRz"(%1, %2, %cst_32) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_33 = arith.constant 0.75540262000000002 : f64
    "qillr.CRy"(%2, %3, %cst_33) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_34 = arith.constant 1.7248285999999999 : f64
    "qillr.CRz"(%2, %3, %cst_34) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_35 = arith.constant 0.50328470000000003 : f64
    "qillr.CRy"(%3, %4, %cst_35) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_36 = arith.constant 2.0834139 : f64
    "qillr.CRz"(%3, %4, %cst_36) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_37 = arith.constant 2.3544627999999999 : f64
    "qillr.CRy"(%4, %5, %cst_37) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_38 = arith.constant 3.6512150999999999 : f64
    "qillr.CRz"(%4, %5, %cst_38) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_39 = arith.constant 2.3064749999999998 : f64
    "qillr.CRy"(%5, %6, %cst_39) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_40 = arith.constant 4.5379702999999996 : f64
    "qillr.CRz"(%5, %6, %cst_40) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_41 = arith.constant 5.9573963000000001 : f64
    "qillr.CRy"(%6, %7, %cst_41) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_42 = arith.constant 1.1291483 : f64
    "qillr.CRz"(%6, %7, %cst_42) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_43 = arith.constant 3.5671268999999999 : f64
    "qillr.CRy"(%7, %8, %cst_43) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_44 = arith.constant 3.3994773 : f64
    "qillr.CRz"(%7, %8, %cst_44) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_45 = arith.constant 4.8900239000000001 : f64
    "qillr.CRy"(%8, %9, %cst_45) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_46 = arith.constant 2.0922298000000001 : f64
    "qillr.CRz"(%8, %9, %cst_46) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_47 = arith.constant 1.9567173 : f64
    "qillr.CRy"(%9, %10, %cst_47) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_48 = arith.constant 3.7551914000000002 : f64
    "qillr.CRz"(%9, %10, %cst_48) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_49 = arith.constant 2.5809011000000002 : f64
    "qillr.CRy"(%10, %11, %cst_49) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_50 = arith.constant 5.9607977999999999 : f64
    "qillr.CRz"(%10, %11, %cst_50) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_51 = arith.constant 3.5095189000000002 : f64
    "qillr.CRy"(%11, %12, %cst_51) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_52 = arith.constant 3.2731878999999999 : f64
    "qillr.CRz"(%11, %12, %cst_52) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_53 = arith.constant 0.22418943 : f64
    "qillr.CRy"(%12, %13, %cst_53) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_54 = arith.constant 5.4432263000000001 : f64
    "qillr.CRz"(%12, %13, %cst_54) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_55 = arith.constant 6.0341126999999997 : f64
    "qillr.CRy"(%13, %14, %cst_55) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_56 = arith.constant 5.5657544000000003 : f64
    "qillr.CRz"(%13, %14, %cst_56) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_57 = arith.constant 3.3907864999999999 : f64
    "qillr.CRy"(%14, %15, %cst_57) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_58 = arith.constant 5.4888443000000002 : f64
    "qillr.CRz"(%14, %15, %cst_58) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_59 = arith.constant 0.73670833000000002 : f64
    "qillr.CRy"(%15, %16, %cst_59) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %cst_60 = arith.constant 4.5942454000000001 : f64
    "qillr.CRz"(%15, %16, %cst_60) : (!qillr.qubit, !qillr.qubit, f64) -> ()
    %17 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_61 = arith.constant 0.32962971000000002 : f64
    "qillr.Ry"(%17, %cst_61) : (!qillr.qubit, f64) -> ()
    %cst_62 = arith.constant 3.0816631999999999 : f64
    "qillr.Rz"(%17, %cst_62) : (!qillr.qubit, f64) -> ()
    %18 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_63 = arith.constant 0.22442694999999999 : f64
    "qillr.Ry"(%18, %cst_63) : (!qillr.qubit, f64) -> ()
    %cst_64 = arith.constant -1.3386051999999999 : f64
    "qillr.Rz"(%18, %cst_64) : (!qillr.qubit, f64) -> ()
    %19 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_65 = arith.constant -1.8779855999999999 : f64
    "qillr.Ry"(%19, %cst_65) : (!qillr.qubit, f64) -> ()
    %cst_66 = arith.constant -2.8367303000000001 : f64
    "qillr.Rz"(%19, %cst_66) : (!qillr.qubit, f64) -> ()
    %20 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_67 = arith.constant 1.7709714000000001 : f64
    "qillr.Ry"(%20, %cst_67) : (!qillr.qubit, f64) -> ()
    %cst_68 = arith.constant -0.58703185000000002 : f64
    "qillr.Rz"(%20, %cst_68) : (!qillr.qubit, f64) -> ()
    %21 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_69 = arith.constant 2.2660222999999999 : f64
    "qillr.Ry"(%21, %cst_69) : (!qillr.qubit, f64) -> ()
    %cst_70 = arith.constant -0.25728645 : f64
    "qillr.Rz"(%21, %cst_70) : (!qillr.qubit, f64) -> ()
    %22 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_71 = arith.constant 1.7599798 : f64
    "qillr.Ry"(%22, %cst_71) : (!qillr.qubit, f64) -> ()
    %cst_72 = arith.constant -2.9292550999999998 : f64
    "qillr.Rz"(%22, %cst_72) : (!qillr.qubit, f64) -> ()
    %23 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_73 = arith.constant -3.0100533 : f64
    "qillr.Ry"(%23, %cst_73) : (!qillr.qubit, f64) -> ()
    %cst_74 = arith.constant 0.60897316999999995 : f64
    "qillr.Rz"(%23, %cst_74) : (!qillr.qubit, f64) -> ()
    %24 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_75 = arith.constant 1.2306531999999999 : f64
    "qillr.Ry"(%24, %cst_75) : (!qillr.qubit, f64) -> ()
    %cst_76 = arith.constant 0.61434179 : f64
    "qillr.Rz"(%24, %cst_76) : (!qillr.qubit, f64) -> ()
    %25 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_77 = arith.constant -1.1195911999999999 : f64
    "qillr.Ry"(%25, %cst_77) : (!qillr.qubit, f64) -> ()
    %cst_78 = arith.constant -1.1137433999999999 : f64
    "qillr.Rz"(%25, %cst_78) : (!qillr.qubit, f64) -> ()
    %26 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_79 = arith.constant 2.0774699000000001 : f64
    "qillr.Ry"(%26, %cst_79) : (!qillr.qubit, f64) -> ()
    %cst_80 = arith.constant 1.9504633 : f64
    "qillr.Rz"(%26, %cst_80) : (!qillr.qubit, f64) -> ()
    %27 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_81 = arith.constant 1.0520065000000001 : f64
    "qillr.Ry"(%27, %cst_81) : (!qillr.qubit, f64) -> ()
    %cst_82 = arith.constant 0.58836451999999995 : f64
    "qillr.Rz"(%27, %cst_82) : (!qillr.qubit, f64) -> ()
    %28 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_83 = arith.constant -2.2668868 : f64
    "qillr.Ry"(%28, %cst_83) : (!qillr.qubit, f64) -> ()
    %cst_84 = arith.constant 1.4253743999999999 : f64
    "qillr.Rz"(%28, %cst_84) : (!qillr.qubit, f64) -> ()
    %29 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_85 = arith.constant 0.28180767000000001 : f64
    "qillr.Ry"(%29, %cst_85) : (!qillr.qubit, f64) -> ()
    %cst_86 = arith.constant 1.7173491000000001 : f64
    "qillr.Rz"(%29, %cst_86) : (!qillr.qubit, f64) -> ()
    %30 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_87 = arith.constant -1.8301518999999999 : f64
    "qillr.Ry"(%30, %cst_87) : (!qillr.qubit, f64) -> ()
    %cst_88 = arith.constant -2.7255153000000001 : f64
    "qillr.Rz"(%30, %cst_88) : (!qillr.qubit, f64) -> ()
    %31 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_89 = arith.constant -0.19044523999999999 : f64
    "qillr.Ry"(%31, %cst_89) : (!qillr.qubit, f64) -> ()
    %cst_90 = arith.constant -2.1166116000000001 : f64
    "qillr.Rz"(%31, %cst_90) : (!qillr.qubit, f64) -> ()
    %32 = "qillr.alloc"() : () -> !qillr.qubit
    %cst_91 = arith.constant 1.6016855000000001 : f64
    "qillr.Ry"(%32, %cst_91) : (!qillr.qubit, f64) -> ()
    %cst_92 = arith.constant 0.93765918999999998 : f64
    "qillr.Rz"(%32, %cst_92) : (!qillr.qubit, f64) -> ()
    "qillr.cswap"(%0, %1, %17) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %2, %18) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %3, %19) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %4, %20) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %5, %21) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %6, %22) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %7, %23) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %8, %24) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %9, %25) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %10, %26) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %11, %27) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %12, %28) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %13, %29) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %14, %30) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %15, %31) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%0, %16, %32) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.H"(%0) : (!qillr.qubit) -> ()
    "qillr.barrier"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32) : (!qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    %33 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%0, %33) : (!qillr.qubit, !qillr.result) -> ()
    %34 = "qillr.read_measurement"(%33) : (!qillr.result) -> i1
    %35 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%1, %35) : (!qillr.qubit, !qillr.result) -> ()
    %36 = "qillr.read_measurement"(%35) : (!qillr.result) -> i1
    %37 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%2, %37) : (!qillr.qubit, !qillr.result) -> ()
    %38 = "qillr.read_measurement"(%37) : (!qillr.result) -> i1
    %39 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%3, %39) : (!qillr.qubit, !qillr.result) -> ()
    %40 = "qillr.read_measurement"(%39) : (!qillr.result) -> i1
    %41 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%4, %41) : (!qillr.qubit, !qillr.result) -> ()
    %42 = "qillr.read_measurement"(%41) : (!qillr.result) -> i1
    %43 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%5, %43) : (!qillr.qubit, !qillr.result) -> ()
    %44 = "qillr.read_measurement"(%43) : (!qillr.result) -> i1
    %45 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%6, %45) : (!qillr.qubit, !qillr.result) -> ()
    %46 = "qillr.read_measurement"(%45) : (!qillr.result) -> i1
    %47 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%7, %47) : (!qillr.qubit, !qillr.result) -> ()
    %48 = "qillr.read_measurement"(%47) : (!qillr.result) -> i1
    %49 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%8, %49) : (!qillr.qubit, !qillr.result) -> ()
    %50 = "qillr.read_measurement"(%49) : (!qillr.result) -> i1
    %51 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%9, %51) : (!qillr.qubit, !qillr.result) -> ()
    %52 = "qillr.read_measurement"(%51) : (!qillr.result) -> i1
    %53 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%10, %53) : (!qillr.qubit, !qillr.result) -> ()
    %54 = "qillr.read_measurement"(%53) : (!qillr.result) -> i1
    %55 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%11, %55) : (!qillr.qubit, !qillr.result) -> ()
    %56 = "qillr.read_measurement"(%55) : (!qillr.result) -> i1
    %57 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%12, %57) : (!qillr.qubit, !qillr.result) -> ()
    %58 = "qillr.read_measurement"(%57) : (!qillr.result) -> i1
    %59 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%13, %59) : (!qillr.qubit, !qillr.result) -> ()
    %60 = "qillr.read_measurement"(%59) : (!qillr.result) -> i1
    %61 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%14, %61) : (!qillr.qubit, !qillr.result) -> ()
    %62 = "qillr.read_measurement"(%61) : (!qillr.result) -> i1
    %63 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%15, %63) : (!qillr.qubit, !qillr.result) -> ()
    %64 = "qillr.read_measurement"(%63) : (!qillr.result) -> i1
    %65 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%16, %65) : (!qillr.qubit, !qillr.result) -> ()
    %66 = "qillr.read_measurement"(%65) : (!qillr.result) -> i1
    %67 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%17, %67) : (!qillr.qubit, !qillr.result) -> ()
    %68 = "qillr.read_measurement"(%67) : (!qillr.result) -> i1
    %69 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%18, %69) : (!qillr.qubit, !qillr.result) -> ()
    %70 = "qillr.read_measurement"(%69) : (!qillr.result) -> i1
    %71 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%19, %71) : (!qillr.qubit, !qillr.result) -> ()
    %72 = "qillr.read_measurement"(%71) : (!qillr.result) -> i1
    %73 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%20, %73) : (!qillr.qubit, !qillr.result) -> ()
    %74 = "qillr.read_measurement"(%73) : (!qillr.result) -> i1
    %75 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%21, %75) : (!qillr.qubit, !qillr.result) -> ()
    %76 = "qillr.read_measurement"(%75) : (!qillr.result) -> i1
    %77 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%22, %77) : (!qillr.qubit, !qillr.result) -> ()
    %78 = "qillr.read_measurement"(%77) : (!qillr.result) -> i1
    %79 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%23, %79) : (!qillr.qubit, !qillr.result) -> ()
    %80 = "qillr.read_measurement"(%79) : (!qillr.result) -> i1
    %81 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%24, %81) : (!qillr.qubit, !qillr.result) -> ()
    %82 = "qillr.read_measurement"(%81) : (!qillr.result) -> i1
    %83 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%25, %83) : (!qillr.qubit, !qillr.result) -> ()
    %84 = "qillr.read_measurement"(%83) : (!qillr.result) -> i1
    %85 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%26, %85) : (!qillr.qubit, !qillr.result) -> ()
    %86 = "qillr.read_measurement"(%85) : (!qillr.result) -> i1
    %87 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%27, %87) : (!qillr.qubit, !qillr.result) -> ()
    %88 = "qillr.read_measurement"(%87) : (!qillr.result) -> i1
    %89 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%28, %89) : (!qillr.qubit, !qillr.result) -> ()
    %90 = "qillr.read_measurement"(%89) : (!qillr.result) -> i1
    %91 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%29, %91) : (!qillr.qubit, !qillr.result) -> ()
    %92 = "qillr.read_measurement"(%91) : (!qillr.result) -> i1
    %93 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%30, %93) : (!qillr.qubit, !qillr.result) -> ()
    %94 = "qillr.read_measurement"(%93) : (!qillr.result) -> i1
    %95 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%31, %95) : (!qillr.qubit, !qillr.result) -> ()
    %96 = "qillr.read_measurement"(%95) : (!qillr.result) -> i1
    %97 = "qillr.ralloc"() : () -> !qillr.result
    "qillr.measure"(%32, %97) : (!qillr.qubit, !qillr.result) -> ()
    %98 = "qillr.read_measurement"(%97) : (!qillr.result) -> i1
    "qillr.reset"(%0) : (!qillr.qubit) -> ()
    "qillr.reset"(%1) : (!qillr.qubit) -> ()
    "qillr.reset"(%2) : (!qillr.qubit) -> ()
    "qillr.reset"(%3) : (!qillr.qubit) -> ()
    "qillr.reset"(%4) : (!qillr.qubit) -> ()
    "qillr.reset"(%5) : (!qillr.qubit) -> ()
    "qillr.reset"(%6) : (!qillr.qubit) -> ()
    "qillr.reset"(%7) : (!qillr.qubit) -> ()
    "qillr.reset"(%8) : (!qillr.qubit) -> ()
    "qillr.reset"(%9) : (!qillr.qubit) -> ()
    "qillr.reset"(%10) : (!qillr.qubit) -> ()
    "qillr.reset"(%11) : (!qillr.qubit) -> ()
    "qillr.reset"(%12) : (!qillr.qubit) -> ()
    "qillr.reset"(%13) : (!qillr.qubit) -> ()
    "qillr.reset"(%14) : (!qillr.qubit) -> ()
    "qillr.reset"(%15) : (!qillr.qubit) -> ()
    "qillr.reset"(%16) : (!qillr.qubit) -> ()
    "qillr.reset"(%17) : (!qillr.qubit) -> ()
    "qillr.reset"(%18) : (!qillr.qubit) -> ()
    "qillr.reset"(%19) : (!qillr.qubit) -> ()
    "qillr.reset"(%20) : (!qillr.qubit) -> ()
    "qillr.reset"(%21) : (!qillr.qubit) -> ()
    "qillr.reset"(%22) : (!qillr.qubit) -> ()
    "qillr.reset"(%23) : (!qillr.qubit) -> ()
    "qillr.reset"(%24) : (!qillr.qubit) -> ()
    "qillr.reset"(%25) : (!qillr.qubit) -> ()
    "qillr.reset"(%26) : (!qillr.qubit) -> ()
    "qillr.reset"(%27) : (!qillr.qubit) -> ()
    "qillr.reset"(%28) : (!qillr.qubit) -> ()
    "qillr.reset"(%29) : (!qillr.qubit) -> ()
    "qillr.reset"(%30) : (!qillr.qubit) -> ()
    "qillr.reset"(%31) : (!qillr.qubit) -> ()
    "qillr.reset"(%32) : (!qillr.qubit) -> ()
    return
  }
}
