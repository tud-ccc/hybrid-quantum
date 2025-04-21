// Test quantum circuit
OPENQASM 3.0;
include "stdgates.inc"; 
qubit[2] q;
bit[2] c;

// Apply all single-qubit gates
h q[0];
x q[0];
rx(3.1415) q[0];

// Two-qubit gates
cx q[0], q[1];
swap q[0], q[1];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];

// Reset qubits
reset q[0];
reset q[1];