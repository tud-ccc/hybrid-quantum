// RUN: quantum-opt %s \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:       convert-quantum-to-qir, \
// RUN:       func.func(convert-scf-to-cf), \
// RUN:       convert-qir-to-llvm, \
// RUN:       convert-func-to-llvm, \
// RUN:       convert-cf-to-llvm, \
// RUN:       convert-vector-to-llvm, \
// RUN:       one-shot-bufferize{allow-unknown-ops}, \
// RUN:       finalize-memref-to-llvm, \
// RUN:       convert-index-to-llvm, \
// RUN:       convert-arith-to-llvm, \
// RUN:       reconcile-unrealized-casts)" | \
// RUN: mlir-runner -e entry -entry-point-result=void \
// RUN:     --shared-libs=%qir_shlibs,%mlir_c_runner_utils | \
// RUN: FileCheck %s --match-full-lines

module {
//QASMbench dnn small example code converted to qir dialect. 
 func.func @dnn_qasm_bench() {
    %0 = arith.constant 1.100000 : f64
    %1 = "qir.alloc" () : () -> (!qir.qubit)
    "qir.Rx" (%1, %0) : (!qir.qubit, f64) -> ()
    %2 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %2) : (!qir.qubit, f64) -> ()
    %3 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %3) : (!qir.qubit, f64) -> ()
    %4 = arith.constant 1.100000 : f64
    %5 = "qir.alloc" () : () -> (!qir.qubit)
    "qir.Rx" (%5, %4) : (!qir.qubit, f64) -> ()
    %6 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %6) : (!qir.qubit, f64) -> ()
    %7 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %7) : (!qir.qubit, f64) -> ()
    %8 = arith.constant 3.455752 : f64
    "qir.Rz" (%1, %8) : (!qir.qubit, f64) -> ()
    %9 = arith.constant 3.455752 : f64
    "qir.Rz" (%5, %9) : (!qir.qubit, f64) -> ()
    %10 = arith.constant 0.000000 : f64
    %11 = arith.constant 1.570796 : f64
    %12 = arith.constant 0.785398 : f64
    "qir.U" (%1, %11, %10, %12) : (!qir.qubit, f64, f64, f64) -> ()
    %13 = arith.constant 3.141593 : f64
    %14 = arith.constant 1.570796 : f64
    %15 = arith.constant 2.356194 : f64
    "qir.U" (%5, %14, %13, %15) : (!qir.qubit, f64, f64, f64) -> ()
    %16 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %16) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %17 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %17) : (!qir.qubit, f64) -> ()
    %18 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %18) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %19 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %19) : (!qir.qubit, f64) -> ()
    %20 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %20) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %21 = arith.constant 2.042035 : f64
    %22 = arith.constant 1.570796 : f64
    %23 = arith.constant 3.141593 : f64
    "qir.U" (%1, %22, %21, %23) : (!qir.qubit, f64, f64, f64) -> ()
    %24 = arith.constant 0.471239 : f64
    %25 = arith.constant 1.570796 : f64
    %26 = arith.constant 0.000000 : f64
    "qir.U" (%5, %25, %24, %26) : (!qir.qubit, f64, f64, f64) -> ()
    %27 = arith.constant 3.141593 : f64
    %28 = arith.constant 0.000000 : f64
    %29 = arith.constant 1.570796 : f64
    "qir.U" (%1, %28, %27, %29) : (!qir.qubit, f64, f64, f64) -> ()
    %30 = arith.constant 0.000000 : f64
    %31 = arith.constant 0.000000 : f64
    %32 = arith.constant 1.570796 : f64
    "qir.U" (%5, %31, %30, %32) : (!qir.qubit, f64, f64, f64) -> ()
    %33 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %33) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %34 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %34) : (!qir.qubit, f64) -> ()
    %35 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %35) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %36 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %36) : (!qir.qubit, f64) -> ()
    %37 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %37) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %38 = arith.constant 0.000000 : f64
    %39 = arith.constant 3.141593 : f64
    %40 = arith.constant 1.570796 : f64
    "qir.U" (%1, %39, %38, %40) : (!qir.qubit, f64, f64, f64) -> ()
    %41 = arith.constant 0.000000 : f64
    %42 = arith.constant 3.141593 : f64
    %43 = arith.constant 4.712389 : f64
    "qir.U" (%5, %42, %41, %43) : (!qir.qubit, f64, f64, f64) -> ()
    %44 = arith.constant 4.712389 : f64
    %45 = arith.constant 1.570796 : f64
    %46 = arith.constant 4.712389 : f64
    "qir.U" (%1, %45, %44, %46) : (!qir.qubit, f64, f64, f64) -> ()
    %47 = arith.constant 1.570796 : f64
    %48 = arith.constant 1.570796 : f64
    %49 = arith.constant 4.712389 : f64
    "qir.U" (%5, %48, %47, %49) : (!qir.qubit, f64, f64, f64) -> ()
    %50 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %50) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %51 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %51) : (!qir.qubit, f64) -> ()
    %52 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %52) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %53 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %53) : (!qir.qubit, f64) -> ()
    %54 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %54) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %55 = arith.constant 1.570796 : f64
    %56 = arith.constant 1.570796 : f64
    %57 = arith.constant 1.570796 : f64
    "qir.U" (%1, %56, %55, %57) : (!qir.qubit, f64, f64, f64) -> ()
    %58 = arith.constant 1.570796 : f64
    %59 = arith.constant 1.570796 : f64
    %60 = arith.constant 4.712389 : f64
    "qir.U" (%5, %59, %58, %60) : (!qir.qubit, f64, f64, f64) -> ()
    %61 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %61) : (!qir.qubit, f64) -> ()
    %62 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %62) : (!qir.qubit, f64) -> ()
    %63 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %63) : (!qir.qubit, f64) -> ()
    %64 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %64) : (!qir.qubit, f64) -> ()
    %65 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %65) : (!qir.qubit, f64) -> ()
    %66 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %66) : (!qir.qubit, f64) -> ()
    %67 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %67) : (!qir.qubit, f64) -> ()
    %68 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %68) : (!qir.qubit, f64) -> ()
    %69 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %69) : (!qir.qubit, f64) -> ()
    %70 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %70) : (!qir.qubit, f64) -> ()
    %71 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %71) : (!qir.qubit, f64) -> ()
    %72 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %72) : (!qir.qubit, f64) -> ()
    %73 = arith.constant 3.455752 : f64
    "qir.Rz" (%5, %73) : (!qir.qubit, f64) -> ()
    %74 = arith.constant 3.455752 : f64
    "qir.Rz" (%1, %74) : (!qir.qubit, f64) -> ()
    %75 = arith.constant 0.000000 : f64
    %76 = arith.constant 1.570796 : f64
    %77 = arith.constant 0.785398 : f64
    "qir.U" (%5, %76, %75, %77) : (!qir.qubit, f64, f64, f64) -> ()
    %78 = arith.constant 3.141593 : f64
    %79 = arith.constant 1.570796 : f64
    %80 = arith.constant 2.356194 : f64
    "qir.U" (%1, %79, %78, %80) : (!qir.qubit, f64, f64, f64) -> ()
    %81 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %81) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %82 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %82) : (!qir.qubit, f64) -> ()
    %83 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %83) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %84 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %84) : (!qir.qubit, f64) -> ()
    %85 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %85) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %86 = arith.constant 2.042035 : f64
    %87 = arith.constant 1.570796 : f64
    %88 = arith.constant 3.141593 : f64
    "qir.U" (%5, %87, %86, %88) : (!qir.qubit, f64, f64, f64) -> ()
    %89 = arith.constant 0.471239 : f64
    %90 = arith.constant 1.570796 : f64
    %91 = arith.constant 0.000000 : f64
    "qir.U" (%1, %90, %89, %91) : (!qir.qubit, f64, f64, f64) -> ()
    %92 = arith.constant 3.141593 : f64
    %93 = arith.constant 0.000000 : f64
    %94 = arith.constant 1.570796 : f64
    "qir.U" (%5, %93, %92, %94) : (!qir.qubit, f64, f64, f64) -> ()
    %95 = arith.constant 0.000000 : f64
    %96 = arith.constant 0.000000 : f64
    %97 = arith.constant 1.570796 : f64
    "qir.U" (%1, %96, %95, %97) : (!qir.qubit, f64, f64, f64) -> ()
    %98 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %98) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %99 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %99) : (!qir.qubit, f64) -> ()
    %100 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %100) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %101 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %101) : (!qir.qubit, f64) -> ()
    %102 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %102) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %103 = arith.constant 0.000000 : f64
    %104 = arith.constant 3.141593 : f64
    %105 = arith.constant 1.570796 : f64
    "qir.U" (%5, %104, %103, %105) : (!qir.qubit, f64, f64, f64) -> ()
    %106 = arith.constant 0.000000 : f64
    %107 = arith.constant 3.141593 : f64
    %108 = arith.constant 4.712389 : f64
    "qir.U" (%1, %107, %106, %108) : (!qir.qubit, f64, f64, f64) -> ()
    %109 = arith.constant 4.712389 : f64
    %110 = arith.constant 1.570796 : f64
    %111 = arith.constant 4.712389 : f64
    "qir.U" (%5, %110, %109, %111) : (!qir.qubit, f64, f64, f64) -> ()
    %112 = arith.constant 1.570796 : f64
    %113 = arith.constant 1.570796 : f64
    %114 = arith.constant 4.712389 : f64
    "qir.U" (%1, %113, %112, %114) : (!qir.qubit, f64, f64, f64) -> ()
    %115 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %115) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %116 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %116) : (!qir.qubit, f64) -> ()
    %117 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %117) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %118 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %118) : (!qir.qubit, f64) -> ()
    %119 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %119) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %120 = arith.constant 1.570796 : f64
    %121 = arith.constant 1.570796 : f64
    %122 = arith.constant 1.570796 : f64
    "qir.U" (%5, %121, %120, %122) : (!qir.qubit, f64, f64, f64) -> ()
    %123 = arith.constant 1.570796 : f64
    %124 = arith.constant 1.570796 : f64
    %125 = arith.constant 4.712389 : f64
    "qir.U" (%1, %124, %123, %125) : (!qir.qubit, f64, f64, f64) -> ()
    %126 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %126) : (!qir.qubit, f64) -> ()
    %127 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %127) : (!qir.qubit, f64) -> ()
    %128 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %128) : (!qir.qubit, f64) -> ()
    %129 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %129) : (!qir.qubit, f64) -> ()
    %130 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %130) : (!qir.qubit, f64) -> ()
    %131 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %131) : (!qir.qubit, f64) -> ()
    %132 = arith.constant -1.570796 : f64
    "qir.Ry" (%5, %132) : (!qir.qubit, f64) -> ()
    %133 = arith.constant 0.000000 : f64
    %134 = arith.constant 1.570796 : f64
    %135 = arith.constant 0.785398 : f64
    "qir.U" (%1, %134, %133, %135) : (!qir.qubit, f64, f64, f64) -> ()
    %136 = arith.constant 3.141593 : f64
    %137 = arith.constant 1.570796 : f64
    %138 = arith.constant 2.356194 : f64
    "qir.U" (%5, %137, %136, %138) : (!qir.qubit, f64, f64, f64) -> ()
    %139 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %139) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %140 = arith.constant 0.157080 : f64
    "qir.Rx" (%1, %140) : (!qir.qubit, f64) -> ()
    %141 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %141) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %142 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %142) : (!qir.qubit, f64) -> ()
    %143 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %143) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %144 = arith.constant 0.942478 : f64
    %145 = arith.constant 1.570796 : f64
    %146 = arith.constant 3.141593 : f64
    "qir.U" (%1, %145, %144, %146) : (!qir.qubit, f64, f64, f64) -> ()
    %147 = arith.constant 5.654867 : f64
    %148 = arith.constant 1.570796 : f64
    %149 = arith.constant 0.000000 : f64
    "qir.U" (%5, %148, %147, %149) : (!qir.qubit, f64, f64, f64) -> ()
    %150 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %150) : (!qir.qubit, f64) -> ()
    %151 = arith.constant 0.000000 : f64
    %152 = arith.constant 1.570796 : f64
    %153 = arith.constant 0.785398 : f64
    "qir.U" (%1, %152, %151, %153) : (!qir.qubit, f64, f64, f64) -> ()
    %154 = arith.constant 3.141593 : f64
    %155 = arith.constant 1.570796 : f64
    %156 = arith.constant 2.356194 : f64
    "qir.U" (%5, %155, %154, %156) : (!qir.qubit, f64, f64, f64) -> ()
    %157 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %157) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %158 = arith.constant 0.157080 : f64
    "qir.Rx" (%1, %158) : (!qir.qubit, f64) -> ()
    %159 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %159) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %160 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %160) : (!qir.qubit, f64) -> ()
    %161 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %161) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %162 = arith.constant 0.942478 : f64
    %163 = arith.constant 1.570796 : f64
    %164 = arith.constant 3.141593 : f64
    "qir.U" (%1, %163, %162, %164) : (!qir.qubit, f64, f64, f64) -> ()
    %165 = arith.constant 5.654867 : f64
    %166 = arith.constant 1.570796 : f64
    %167 = arith.constant 0.000000 : f64
    "qir.U" (%5, %166, %165, %167) : (!qir.qubit, f64, f64, f64) -> ()
    %168 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %168) : (!qir.qubit, f64) -> ()
    %169 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %169) : (!qir.qubit, f64) -> ()
    %170 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %170) : (!qir.qubit, f64) -> ()
    %171 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %171) : (!qir.qubit, f64) -> ()
    %172 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %172) : (!qir.qubit, f64) -> ()
    %173 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %173) : (!qir.qubit, f64) -> ()
    %174 = arith.constant 3.455752 : f64
    "qir.Rz" (%1, %174) : (!qir.qubit, f64) -> ()
    %175 = arith.constant 3.455752 : f64
    "qir.Rz" (%5, %175) : (!qir.qubit, f64) -> ()
    %176 = arith.constant 0.000000 : f64
    %177 = arith.constant 1.570796 : f64
    %178 = arith.constant 0.785398 : f64
    "qir.U" (%1, %177, %176, %178) : (!qir.qubit, f64, f64, f64) -> ()
    %179 = arith.constant 3.141593 : f64
    %180 = arith.constant 1.570796 : f64
    %181 = arith.constant 2.356194 : f64
    "qir.U" (%5, %180, %179, %181) : (!qir.qubit, f64, f64, f64) -> ()
    %182 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %182) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %183 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %183) : (!qir.qubit, f64) -> ()
    %184 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %184) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %185 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %185) : (!qir.qubit, f64) -> ()
    %186 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %186) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %187 = arith.constant 2.042035 : f64
    %188 = arith.constant 1.570796 : f64
    %189 = arith.constant 3.141593 : f64
    "qir.U" (%1, %188, %187, %189) : (!qir.qubit, f64, f64, f64) -> ()
    %190 = arith.constant 0.471239 : f64
    %191 = arith.constant 1.570796 : f64
    %192 = arith.constant 0.000000 : f64
    "qir.U" (%5, %191, %190, %192) : (!qir.qubit, f64, f64, f64) -> ()
    %193 = arith.constant 3.141593 : f64
    %194 = arith.constant 0.000000 : f64
    %195 = arith.constant 1.570796 : f64
    "qir.U" (%1, %194, %193, %195) : (!qir.qubit, f64, f64, f64) -> ()
    %196 = arith.constant 0.000000 : f64
    %197 = arith.constant 0.000000 : f64
    %198 = arith.constant 1.570796 : f64
    "qir.U" (%5, %197, %196, %198) : (!qir.qubit, f64, f64, f64) -> ()
    %199 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %199) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %200 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %200) : (!qir.qubit, f64) -> ()
    %201 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %201) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %202 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %202) : (!qir.qubit, f64) -> ()
    %203 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %203) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %204 = arith.constant 0.000000 : f64
    %205 = arith.constant 3.141593 : f64
    %206 = arith.constant 1.570796 : f64
    "qir.U" (%1, %205, %204, %206) : (!qir.qubit, f64, f64, f64) -> ()
    %207 = arith.constant 0.000000 : f64
    %208 = arith.constant 3.141593 : f64
    %209 = arith.constant 4.712389 : f64
    "qir.U" (%5, %208, %207, %209) : (!qir.qubit, f64, f64, f64) -> ()
    %210 = arith.constant 4.712389 : f64
    %211 = arith.constant 1.570796 : f64
    %212 = arith.constant 4.712389 : f64
    "qir.U" (%1, %211, %210, %212) : (!qir.qubit, f64, f64, f64) -> ()
    %213 = arith.constant 1.570796 : f64
    %214 = arith.constant 1.570796 : f64
    %215 = arith.constant 4.712389 : f64
    "qir.U" (%5, %214, %213, %215) : (!qir.qubit, f64, f64, f64) -> ()
    %216 = arith.constant 1.570796 : f64
    "qir.Rx" (%1, %216) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %217 = arith.constant 1.256637 : f64
    "qir.Rx" (%1, %217) : (!qir.qubit, f64) -> ()
    %218 = arith.constant 1.570796 : f64
    "qir.Ry" (%5, %218) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %219 = arith.constant -1.570796 : f64
    "qir.Rx" (%5, %219) : (!qir.qubit, f64) -> ()
    %220 = arith.constant 1.570796 : f64
    "qir.Rz" (%5, %220) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %221 = arith.constant 1.570796 : f64
    %222 = arith.constant 1.570796 : f64
    %223 = arith.constant 1.570796 : f64
    "qir.U" (%1, %222, %221, %223) : (!qir.qubit, f64, f64, f64) -> ()
    %224 = arith.constant 1.570796 : f64
    %225 = arith.constant 1.570796 : f64
    %226 = arith.constant 4.712389 : f64
    "qir.U" (%5, %225, %224, %226) : (!qir.qubit, f64, f64, f64) -> ()
    %227 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %227) : (!qir.qubit, f64) -> ()
    %228 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %228) : (!qir.qubit, f64) -> ()
    %229 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %229) : (!qir.qubit, f64) -> ()
    %230 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %230) : (!qir.qubit, f64) -> ()
    %231 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %231) : (!qir.qubit, f64) -> ()
    %232 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %232) : (!qir.qubit, f64) -> ()
    %233 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %233) : (!qir.qubit, f64) -> ()
    %234 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %234) : (!qir.qubit, f64) -> ()
    %235 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %235) : (!qir.qubit, f64) -> ()
    %236 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %236) : (!qir.qubit, f64) -> ()
    %237 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %237) : (!qir.qubit, f64) -> ()
    %238 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %238) : (!qir.qubit, f64) -> ()
    %239 = arith.constant 3.455752 : f64
    "qir.Rz" (%5, %239) : (!qir.qubit, f64) -> ()
    %240 = arith.constant 3.455752 : f64
    "qir.Rz" (%1, %240) : (!qir.qubit, f64) -> ()
    %241 = arith.constant 0.000000 : f64
    %242 = arith.constant 1.570796 : f64
    %243 = arith.constant 0.785398 : f64
    "qir.U" (%5, %242, %241, %243) : (!qir.qubit, f64, f64, f64) -> ()
    %244 = arith.constant 3.141593 : f64
    %245 = arith.constant 1.570796 : f64
    %246 = arith.constant 2.356194 : f64
    "qir.U" (%1, %245, %244, %246) : (!qir.qubit, f64, f64, f64) -> ()
    %247 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %247) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %248 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %248) : (!qir.qubit, f64) -> ()
    %249 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %249) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %250 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %250) : (!qir.qubit, f64) -> ()
    %251 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %251) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %252 = arith.constant 2.042035 : f64
    %253 = arith.constant 1.570796 : f64
    %254 = arith.constant 3.141593 : f64
    "qir.U" (%5, %253, %252, %254) : (!qir.qubit, f64, f64, f64) -> ()
    %255 = arith.constant 0.471239 : f64
    %256 = arith.constant 1.570796 : f64
    %257 = arith.constant 0.000000 : f64
    "qir.U" (%1, %256, %255, %257) : (!qir.qubit, f64, f64, f64) -> ()
    %258 = arith.constant 3.141593 : f64
    %259 = arith.constant 0.000000 : f64
    %260 = arith.constant 1.570796 : f64
    "qir.U" (%5, %259, %258, %260) : (!qir.qubit, f64, f64, f64) -> ()
    %261 = arith.constant 0.000000 : f64
    %262 = arith.constant 0.000000 : f64
    %263 = arith.constant 1.570796 : f64
    "qir.U" (%1, %262, %261, %263) : (!qir.qubit, f64, f64, f64) -> ()
    %264 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %264) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %265 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %265) : (!qir.qubit, f64) -> ()
    %266 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %266) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %267 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %267) : (!qir.qubit, f64) -> ()
    %268 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %268) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %269 = arith.constant 0.000000 : f64
    %270 = arith.constant 3.141593 : f64
    %271 = arith.constant 1.570796 : f64
    "qir.U" (%5, %270, %269, %271) : (!qir.qubit, f64, f64, f64) -> ()
    %272 = arith.constant 0.000000 : f64
    %273 = arith.constant 3.141593 : f64
    %274 = arith.constant 4.712389 : f64
    "qir.U" (%1, %273, %272, %274) : (!qir.qubit, f64, f64, f64) -> ()
    %275 = arith.constant 4.712389 : f64
    %276 = arith.constant 1.570796 : f64
    %277 = arith.constant 4.712389 : f64
    "qir.U" (%5, %276, %275, %277) : (!qir.qubit, f64, f64, f64) -> ()
    %278 = arith.constant 1.570796 : f64
    %279 = arith.constant 1.570796 : f64
    %280 = arith.constant 4.712389 : f64
    "qir.U" (%1, %279, %278, %280) : (!qir.qubit, f64, f64, f64) -> ()
    %281 = arith.constant 1.570796 : f64
    "qir.Rx" (%5, %281) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %282 = arith.constant 1.256637 : f64
    "qir.Rx" (%5, %282) : (!qir.qubit, f64) -> ()
    %283 = arith.constant 1.570796 : f64
    "qir.Ry" (%1, %283) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%1, %5) : (!qir.qubit, !qir.qubit) -> ()
    %284 = arith.constant -1.570796 : f64
    "qir.Rx" (%1, %284) : (!qir.qubit, f64) -> ()
    %285 = arith.constant 1.570796 : f64
    "qir.Rz" (%1, %285) : (!qir.qubit, f64) -> ()
    "qir.CNOT" (%5, %1) : (!qir.qubit, !qir.qubit) -> ()
    %286 = arith.constant 1.570796 : f64
    %287 = arith.constant 1.570796 : f64
    %288 = arith.constant 1.570796 : f64
    "qir.U" (%5, %287, %286, %288) : (!qir.qubit, f64, f64, f64) -> ()
    %289 = arith.constant 1.570796 : f64
    %290 = arith.constant 1.570796 : f64
    %291 = arith.constant 4.712389 : f64
    "qir.U" (%1, %290, %289, %291) : (!qir.qubit, f64, f64, f64) -> ()
    %292 = arith.constant 1.100000 : f64
    "qir.Rx" (%5, %292) : (!qir.qubit, f64) -> ()
    %293 = arith.constant 1.100000 : f64
    "qir.Ry" (%5, %293) : (!qir.qubit, f64) -> ()
    %294 = arith.constant 1.100000 : f64
    "qir.Rz" (%5, %294) : (!qir.qubit, f64) -> ()
    %295 = arith.constant 1.100000 : f64
    "qir.Rx" (%1, %295) : (!qir.qubit, f64) -> ()
    %296 = arith.constant 1.100000 : f64
    "qir.Ry" (%1, %296) : (!qir.qubit, f64) -> ()
    %297 = arith.constant 1.100000 : f64
    "qir.Rz" (%1, %297) : (!qir.qubit, f64) -> ()
    %298 = "qir.ralloc" () : () -> (!qir.result)
    "qir.measure" (%1, %298) : (!qir.qubit, !qir.result) -> ()
    %299 = "qir.read_measurement" (%298) : (!qir.result) -> (tensor<1xi1>)
    %300 = "qir.ralloc" () : () -> (!qir.result)
    "qir.measure" (%5, %300) : (!qir.qubit, !qir.result) -> ()
    %301 = "qir.read_measurement" (%300) : (!qir.result) -> (tensor<1xi1>)
   
    %i = "index.constant" () {value = 0 : index} : () -> (index)
    %m1 = "tensor.extract" (%299, %i) : (tensor<1xi1>, index) -> (i1)
    %m2 = "tensor.extract" (%301, %i) : (tensor<1xi1>, index) -> (i1)
    vector.print %m1 : i1
    vector.print %m2 : i1
    return
  }

    func.func @entry() {
    // CHECK: {{0|1}}
    // CHECK: {{0|1}}
    func.call @dnn_qasm_bench() : () -> ()
    return
  }
}
