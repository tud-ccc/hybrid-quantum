/// # Sample
/// Bell States
///
/// # Description
/// Bell states or EPR pairs are specific quantum states of two qubits
/// that represent the simplest (and maximal) examples of quantum entanglement.
///
/// This Q# program implements the four different Bell states.
import Std.Diagnostics.*;

operation Main() : (Result, Result)[] {
    // This array contains a label and a preparation operation for each one
    // of the four Bell states.
    let bellStateTuples = [
        ("|Φ+〉", PreparePhiPlus),
        ("|Φ-〉", PreparePhiMinus),
        ("|Ψ+〉", PreparePsiPlus),
        ("|Ψ-〉", PreparePsiMinus)
    ];

    // Prepare all Bell states, show them using the `DumpMachine` operation
    // and measure the Bell state qubits.
    mutable measurements = [];
    for (label, prepare) in bellStateTuples {
        // Allocate the two qubits that will be used to create a Bell state.
        use register = Qubit[2];
        prepare(register);
        Message($"Bell state {label}:");
        DumpMachine();
        set measurements += [(MResetZ(register[0]), MResetZ(register[1]))];
    }
    return measurements;
}

/// # Summary
/// Prepares |Φ+⟩ = (|00⟩+|11⟩)/√2 state assuming `register` is in |00⟩ state.
operation PreparePhiPlus(register : Qubit[]) : Unit {
    H(register[0]);                 // |+0〉
    CNOT(register[0], register[1]); // 1/sqrt(2)(|00〉 + |11〉)
}

/// # Summary
/// Prepares |Φ−⟩ = (|00⟩-|11⟩)/√2 state assuming `register` is in |00⟩ state.
operation PreparePhiMinus(register : Qubit[]) : Unit {
    H(register[0]);                 // |+0〉
    Z(register[0]);                 // |-0〉
    CNOT(register[0], register[1]); // 1/sqrt(2)(|00〉 - |11〉)
}

/// # Summary
/// Prepares |Ψ+⟩ = (|01⟩+|10⟩)/√2 state assuming `register` is in |00⟩ state.
operation PreparePsiPlus(register : Qubit[]) : Unit {
    H(register[0]);                 // |+0〉
    X(register[1]);                 // |+1〉
    CNOT(register[0], register[1]); // 1/sqrt(2)(|01〉 + |10〉)
}

/// # Summary
/// Prepares |Ψ−⟩ = (|01⟩-|10⟩)/√2 state assuming `register` is in |00⟩ state.
operation PreparePsiMinus(register : Qubit[]) : Unit {
    H(register[0]);                 // |+0〉
    Z(register[0]);                 // |-0〉
    X(register[1]);                 // |-1〉
    CNOT(register[0], register[1]); // 1/sqrt(2)(|01〉 - |10〉)
}