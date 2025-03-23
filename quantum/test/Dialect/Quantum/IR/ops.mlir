// RUN: quantum-opt %s | FileCheck %s

%f1 = arith.constant 1.0 : f32  

// CHECK: alloc
%reg = "quantum.alloc" () : () -> (!quantum.qubit<3>)

// Split 
%q0, %q1 = "quantum.split" (%reg) : (!quantum.qubit<3>) -> (!quantum.qubit<2>, !quantum.qubit<1>)
%q00, %q01 = "quantum.split" (%q0) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

// Region with Rz ops
%q00_r, %q01_r = "quantum.region"(%q00, %q01) ({
  ^bb0(%a: !quantum.qubit<1>, %b: !quantum.qubit<1>):

    %theta_a = arith.constant 0.5 : f32
    %theta_b = arith.constant 1.5 : f32

    %a_rz = "quantum.Rz"(%a, %theta_a) : (!quantum.qubit<1>, f32) -> (!quantum.qubit<1>)
    %b_rz = "quantum.Rz"(%b, %theta_b) : (!quantum.qubit<1>, f32) -> (!quantum.qubit<1>)

    "quantum.yield"(%a_rz, %b_rz) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
}) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

// Merge
%lhs = "quantum.alloc" () : () -> !quantum.qubit<3>
%rhs = "quantum.alloc" () : () -> !quantum.qubit<2>
%q = "quantum.merge" (%lhs, %rhs) : (!quantum.qubit<3>, !quantum.qubit<2>) -> (!quantum.qubit<5>)

// Measurements
%qm_single = "quantum.alloc" () : () -> (!quantum.qubit<1>)
%m, %qm_single_out = "quantum.measure_single" (%qm_single) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)

%qm = "quantum.alloc" () : () -> (!quantum.qubit<2>)
%mt, %qm_out = "quantum.measure" (%qm) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)

// Pauli X Y Z, H gates
%q0_X = "quantum.X" (%q00) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_Y = "quantum.Y" (%q0_X) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_Z = "quantum.Z" (%q0_Y) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_H = "quantum.H" (%q0_Z) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
