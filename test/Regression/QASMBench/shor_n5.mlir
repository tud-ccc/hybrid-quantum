// RUN: quantum-opt %s -replace-repeated-reads  | FileCheck %s
// RUN: quantum-opt %s -replace-repeated-reads -lift-qillr-to-quantum -hoist-load-store -eliminate-load-store | FileCheck %s --check-prefix="CHECK2"
// 
// --debug --mlir-print-ir-after-all

module {
  // CHECK-LABEL: @qasm_main
  // CHECK2-LABEL: @qasm_main
  func.func public @qasm_main() {
    // CHECK: %[[Q0:.+]] = "qillr.alloc"() : () -> !qillr.qubit
    %0 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.X"(%0) : (!qillr.qubit) -> ()
    // CHECK: %[[Q1:.+]] = "qillr.alloc"() : () -> !qillr.qubit
    %1 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    // CHECK: %[[R1:.+]] = "qillr.ralloc"() : () -> !qillr.result
    %2 = "qillr.ralloc"() : () -> !qillr.result
    // CHECK: "qillr.measure"(%[[Q1]], %[[R1]]) : (!qillr.qubit, !qillr.result) -> ()
    "qillr.measure"(%1, %2) : (!qillr.qubit, !qillr.result) -> ()
    // CHECK: %[[M1:.+]] = "qillr.read_measurement"(%[[R1]]) : (!qillr.result) -> i1
    %3 = "qillr.read_measurement"(%2) : (!qillr.result) -> i1
    // CHECK :"qillr.reset"(%[[Q1]]) : (!qillr.qubit) -> ()
    "qillr.reset"(%1) : (!qillr.qubit) -> ()
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    %4 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.CNOT"(%1, %4) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%1, %0) : (!qillr.qubit, !qillr.qubit) -> ()
    %5 = "qillr.ralloc"() : () -> !qillr.result
    %6 = "qillr.ralloc"() : () -> !qillr.result
    %7 = "qillr.ralloc"() : () -> !qillr.result
    %8 = "qillr.ralloc"() : () -> !qillr.result
    // CHECK-NOT: %[[M11:.+]] = "qillr.read_measurement"(%[[R1]]) : (!qillr.result) -> i1
    %9 = "qillr.read_measurement"(%2) : (!qillr.result) -> i1
    %true = arith.constant true
    %10 = arith.cmpi eq, %9, %true : i1
    %11 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %false = arith.constant false
    %12 = arith.cmpi eq, %11, %false : i1
    %13 = "qillr.read_measurement"(%6) : (!qillr.result) -> i1
    %false_0 = arith.constant false
    %14 = arith.cmpi eq, %13, %false_0 : i1
    %15 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %false_1 = arith.constant false
    %16 = arith.cmpi eq, %15, %false_1 : i1
    %17 = "qillr.read_measurement"(%8) : (!qillr.result) -> i1
    %false_2 = arith.constant false
    %18 = arith.cmpi eq, %17, %false_2 : i1
    %19 = arith.andi %10, %12 : i1
    %20 = arith.andi %19, %14 : i1
    %21 = arith.andi %20, %16 : i1
    %22 = arith.andi %21, %18 : i1
    scf.if %22 {
      %cst = arith.constant 1.5707963267948966 : f64
      "qillr.U1"(%1, %cst) : (!qillr.qubit, f64) -> ()
    }
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    "qillr.measure"(%1, %5) : (!qillr.qubit, !qillr.result) -> ()
    %23 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    "qillr.reset"(%1) : (!qillr.qubit) -> ()
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    %24 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.cswap"(%1, %24, %0) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.cswap"(%1, %4, %24) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    %25 = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.cswap"(%1, %25, %4) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%1, %25) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%1, %4) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%1, %24) : (!qillr.qubit, !qillr.qubit) -> ()
    "qillr.CNOT"(%1, %0) : (!qillr.qubit, !qillr.qubit) -> ()
    // CHECK-NOT: %[[M12:.+]] = "qillr.read_measurement"(%[[R1]]) : (!qillr.result) -> i1
    %26 = "qillr.read_measurement"(%2) : (!qillr.result) -> i1
    %true_3 = arith.constant true
    %27 = arith.cmpi eq, %26, %true_3 : i1
    %28 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %true_4 = arith.constant true
    %29 = arith.cmpi eq, %28, %true_4 : i1
    %30 = "qillr.read_measurement"(%6) : (!qillr.result) -> i1
    %false_5 = arith.constant false
    %31 = arith.cmpi eq, %30, %false_5 : i1
    %32 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %false_6 = arith.constant false
    %33 = arith.cmpi eq, %32, %false_6 : i1
    %34 = "qillr.read_measurement"(%8) : (!qillr.result) -> i1
    %false_7 = arith.constant false
    %35 = arith.cmpi eq, %34, %false_7 : i1
    %36 = arith.andi %27, %29 : i1
    %37 = arith.andi %36, %31 : i1
    %38 = arith.andi %37, %33 : i1
    %39 = arith.andi %38, %35 : i1
    scf.if %39 {
      %cst = arith.constant 2.3561944901923448 : f64
      "qillr.U1"(%1, %cst) : (!qillr.qubit, f64) -> ()
    }
    // CHECK-NOT: %[[M13:.+]] = "qillr.read_measurement"(%[[R1]]) : (!qillr.result) -> i1
    %40 = "qillr.read_measurement"(%2) : (!qillr.result) -> i1
    %false_8 = arith.constant false
    // CHECK: arith.cmpi eq, %[[M1]], %false : i1
    %41 = arith.cmpi eq, %40, %false_8 : i1
    %42 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %true_9 = arith.constant true
    %43 = arith.cmpi eq, %42, %true_9 : i1
    %44 = "qillr.read_measurement"(%6) : (!qillr.result) -> i1
    %false_10 = arith.constant false
    %45 = arith.cmpi eq, %44, %false_10 : i1
    %46 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %false_11 = arith.constant false
    %47 = arith.cmpi eq, %46, %false_11 : i1
    %48 = "qillr.read_measurement"(%8) : (!qillr.result) -> i1
    %false_12 = arith.constant false
    %49 = arith.cmpi eq, %48, %false_12 : i1
    %50 = arith.andi %41, %43 : i1
    %51 = arith.andi %50, %45 : i1
    %52 = arith.andi %51, %47 : i1
    %53 = arith.andi %52, %49 : i1
    scf.if %53 {
      %cst = arith.constant 1.5707963267948966 : f64
      "qillr.U1"(%1, %cst) : (!qillr.qubit, f64) -> ()
    }
    %54 = "qillr.read_measurement"(%2) : (!qillr.result) -> i1
    %true_13 = arith.constant true
    %55 = arith.cmpi eq, %54, %true_13 : i1
    %56 = "qillr.read_measurement"(%5) : (!qillr.result) -> i1
    %false_14 = arith.constant false
    %57 = arith.cmpi eq, %56, %false_14 : i1
    %58 = "qillr.read_measurement"(%6) : (!qillr.result) -> i1
    %false_15 = arith.constant false
    %59 = arith.cmpi eq, %58, %false_15 : i1
    %60 = "qillr.read_measurement"(%7) : (!qillr.result) -> i1
    %false_16 = arith.constant false
    %61 = arith.cmpi eq, %60, %false_16 : i1
    %62 = "qillr.read_measurement"(%8) : (!qillr.result) -> i1
    %false_17 = arith.constant false
    %63 = arith.cmpi eq, %62, %false_17 : i1
    %64 = arith.andi %55, %57 : i1
    %65 = arith.andi %64, %59 : i1
    %66 = arith.andi %65, %61 : i1
    %67 = arith.andi %66, %63 : i1
    scf.if %67 {
      %cst = arith.constant 0.78539816339744828 : f64
      "qillr.U1"(%1, %cst) : (!qillr.qubit, f64) -> ()
    }
    "qillr.H"(%1) : (!qillr.qubit) -> ()
    "qillr.measure"(%1, %6) : (!qillr.qubit, !qillr.result) -> ()
    %68 = "qillr.read_measurement"(%6) : (!qillr.result) -> i1
    "qillr.reset"(%0) : (!qillr.qubit) -> ()
    "qillr.reset"(%24) : (!qillr.qubit) -> ()
    "qillr.reset"(%4) : (!qillr.qubit) -> ()
    "qillr.reset"(%25) : (!qillr.qubit) -> ()
    return
  }
}
