module {
  func.func @test_quantum_ops() {

    //allocate a tensor
    // %t = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
    %s = quantum.allocate(1) : !quantum.nqubit
    %q = quantum.extract %s[0] : !quantum.nqubit -> !quantum.qubit

    // Define constants
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32

    // Allocate qubits
    %qarray = quantum.allocate(3) :!quantum.nqubit
    %q0 = quantum.extract %qarray[0] : !quantum.nqubit -> !quantum.qubit
    %q1 = quantum.extract %qarray[1] : !quantum.nqubit -> !quantum.qubit
    %q2 = quantum.extract %qarray[2] : !quantum.nqubit -> !quantum.qubit

    //%matmulop = quantum.matmul %qarray, %qarray -> !quantum.nqubit

    %q0_x = quantum.X %q0 : !quantum.qubit
    %q0_y = quantum.Y %q0 : !quantum.qubit
    %q0_z = quantum.Z %q0 : !quantum.qubit
    %q0_h = quantum.H %q0 : !quantum.qubit

    
    //Observaables and Hamiltonians
    %PauliXobs = quantum.namedobs %q[ %c1] : !quantum.obs
    %expectationval = quantum.expectation %PauliXobs : f32

    // Rotation gates
    %q0_rx = quantum.R(%q0, x, %f1) 
    %q0_ry = quantum.R(%q0, x, %f2) 
    %q0_rz = quantum.R(%q0, x, %f1) 

    // Universal single-qubit gate
    %q0_u = quantum.U %q0 (0.1, 0.2, 0.3)
    // Two-qubit gates
    %q0_q1_cnot:2 = quantum.CNOT %q0, %q1
    %q0_q1_cy:2 = quantum.CY %q0, %q1 
    %q0_q1_cz:2 = quantum.CZ %q0, %q1 
    %q0_q1_swap:2 = quantum.SWAP %q0, %q1 

    // Three-qubit gates
    %q0_q1_q2_ccx:3 = quantum.CCX %q0, %q1, %q2 
    %q0_q1_q2_ccz:3 = quantum.CCZ %q0, %q1, %q2 
    %q0_q1_q2_cswap:3 = quantum.CSWAP %q0, %q1, %q2 

    // Measurement
    %res, %m0 = quantum.measure %q0 

    // Deallocate qubits
    quantum.deallocate %qarray : !quantum.nqubit
    return
  }
}