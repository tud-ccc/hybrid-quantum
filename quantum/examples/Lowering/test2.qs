import Microsoft.Quantum.Diagnostics.*;

operation Main() : (Result, Result) {  
    // Allocate two qubits, q1 and q2, in the 0 state.
    use (q1, q2) = (Qubit(), Qubit());
    use encodedRegister = Qubit[10];

    // Put q1 into an even superposition.
    // It now has a 50% chance of being measured as 0 or 1.
    H(encodedRegister[0]);
    CNOT(encodedRegister[0], encodedRegister[1]);
    // Entangle q1 and q2, making q2 depend on q1.

    // Show the entangled state of the qubits.
    DumpMachine();

    // Measure q1 and q2 and store the results in m1 and m2.
    let (m1, m2) = (M(encodedRegister[0]), M(encodedRegister[1]));
    
    // Reset q1 and q2 to the 0 state.
    Reset(encodedRegister[0]);
    Reset(encodedRegister[1]);
    
    // Return the measurement results.
    return (m1, m2);
}