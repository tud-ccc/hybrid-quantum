// RUN: quantum-opt %s | FileCheck %s

//%c1 = arith.constant 1 : index
//%c2 = arith.constant 2 : index
%f1 = arith.constant 1.0 : f32
//%f2 = arith.constant 2.0 : f32
//%f3 = arith.constant 1.5 : f32  

// CHECK: alloc
%reg = "quantum.alloc" () : () -> (!quantum.qubit<3>)

//Split %reg into %q0#0 and %q0#1 of size 2 and 1 respectively. 
%q0, %q1 = "quantum.extract" (%reg) : (!quantum.qubit<3>) -> (!quantum.qubit<2>, !quantum.qubit<1>)
%q00, %q01 = "quantum.extract" (%q0) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

//pauli X Y Z, H gates.
%q0_X = "quantum.X" (%q00) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_Y = "quantum.Y" (%q0_X) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_Z = "quantum.Z" (%q0_Y) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
%q0_H = "quantum.H" (%q0_Z) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)

// Rotation gates.
%reg2 = "quantum.alloc" () : () -> (!quantum.qubit<3>)
//%reg2_RX = "quantum.R" (x, %f1, %reg2) : (!quantum.QuantumAxisAttr, !f32, !quantum.qubit<3>) -> (!quantum.qubit<3>)
//%q0_ry = quantum.R (%q0#0, x, %f2) : !quantum.qubit<2> -> !quantum.qubit<2>
//%q0_rz = quantum.R (%q0#0, x, %f1) : !quantum.qubit<2> -> !quantum.qubit<2>

// Universal single-qubit gate
//%q0_u = quantum.U %q0(%f1, %f2, %f3): !quantum.qubit<2>, f32, f32, f32 -> !quantum.qubit<2>

// Two-qubit gates
// %q0_q1_cnot:2 = quantum.CNOT %q0_x, %q0_y : !quantum.qubit<1>, !quantum.qubit<1> -> (!quantum.qubit<1>, !quantum.qubit<1>)
//%q0_cnot, q1_cnot = "quantum.CNOT" <> (%q0_x, %q0_y) () {} : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)

//%q0_q1_cy:2   = quantum.CY %q0_x, %q0_y   : !quantum.qubit<1>, !quantum.qubit<1> -> !quantum.qubit<1>, !quantum.qubit<1>
//%q0_q1_cz:2   = quantum.CZ %q0_x, %q0_y   : !quantum.qubit<1>, !quantum.qubit<1> -> !quantum.qubit<1>, !quantum.qubit<1>
//%q0_q1_swap:2 = quantum.SWAP %q0_x, %q0_y : !quantum.qubit<1>, !quantum.qubit<1> -> !quantum.qubit<1>, !quantum.qubit<1>

// Three-qubit gates
//%q0_q1_q2_ccx:3 = quantum.CCX %q0_x, %q0_y, %q0_z  : !quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>
//%q0_q1_q2_ccz:3 = quantum.CCZ %q0_x, %q0_y, %q0_z:  !quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>
//%q0_q1_q2_cswap:3 = quantum.CSWAP %q0_x, %q0_y, %q0_z   : !quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>  

// Measurement
//%3:2 = quantum.measure %q0   : !quantum.qubit<2> -> !quantum.qubit<2>

// Deallocate qubits
//quantum.deallocate %reg : !quantum.qubit<3>