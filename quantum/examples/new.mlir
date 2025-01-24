//Test for all quantum operations from the dialect

module {
  func.func @test_quantum_ops() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 1.5 : f32  
    //allocate a  quantum register
    %reg = quantum.alloc : !quantum.qubit<3>

    //operations on register
    %q0 = quantum.extract %reg[%c2] : !quantum.qubit<3> -> !quantum.qubit<2>
    %q0_x = quantum.X %q0 : !quantum.qubit<2> 
    %q0_y = quantum.Y %q0 : !quantum.qubit<2>
    %q0_z = quantum.Z %q0 : !quantum.qubit<2>
    %q0_h = quantum.H %q0 : !quantum.qubit<2>

    // Rotation gates
    %q0_rx = quantum.R (%q0, x, %f1) : !quantum.qubit<2> -> !quantum.qubit<2>
    %q0_ry = quantum.R (%q0, x, %f2) : !quantum.qubit<2> -> !quantum.qubit<2>
    %q0_rz = quantum.R (%q0, x, %f1) : !quantum.qubit<2> -> !quantum.qubit<2>

    // Universal single-qubit gate
    %q0_u = quantum.U %q0(%f1, %f2, %f3): !quantum.qubit<2>, f32, f32, f32 -> !quantum.qubit<2>

    // Two-qubit gates
    %q0_q1_cnot:2 = quantum.CNOT %q0_x, %q0_y : !quantum.qubit<2>, !quantum.qubit<2> -> !quantum.qubit<2>, !quantum.qubit<2>
    %q0_q1_cy:2   = quantum.CY %q0_x, %q0_y   : !quantum.qubit<2>, !quantum.qubit<2> -> !quantum.qubit<2>, !quantum.qubit<2>
    %q0_q1_cz:2   = quantum.CZ %q0_x, %q0_y   : !quantum.qubit<2>, !quantum.qubit<2> -> !quantum.qubit<2>, !quantum.qubit<2>
    %q0_q1_swap:2 = quantum.SWAP %q0_x, %q0_y : !quantum.qubit<2>, !quantum.qubit<2> -> !quantum.qubit<2>, !quantum.qubit<2>

    // Three-qubit gates
    %q0_q1_q2_ccx:3 = quantum.CCX %q0_x, %q0_y, %q0_z  : !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2>  -> !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2> 
    %q0_q1_q2_ccz:3 = quantum.CCZ %q0_x, %q0_y, %q0_z    : !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2>  -> !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2> 
    %q0_q1_q2_cswap:3 = quantum.CSWAP %q0_x, %q0_y, %q0_z   : !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2>  -> !quantum.qubit<2>, !quantum.qubit<2>, !quantum.qubit<2> 

    // Measurement
    %3:2 = quantum.measure %q0   : !quantum.qubit<2> -> !quantum.qubit<2>

    // Deallocate qubits
    quantum.deallocate %reg : !quantum.qubit<3>

    return
  }
}