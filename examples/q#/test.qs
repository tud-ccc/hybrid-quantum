namespace VQEAdaptiveExample {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Arrays;

    /// # Summary
    /// Prepares a parameterized ansatz state on 2 qubits.
    /// This circuit is static (its structure is fixed) and takes the parameters as input.
    operation PrepareAnsatz(qs : Qubit[], parameters : Double[]) : Unit {
        // Example: Apply an Rz rotation on qubit 0, then a CNOT, then an Rz rotation on qubit 1.
        Rz(parameters[0], qs[0]);
        CNOT(qs[0], qs[1]);
        Rz(parameters[1], qs[1]);
    }

    /// # Summary
    /// Runs the quantum circuit for a fixed number of shots and returns the total number of Zero outcomes.
    /// The measurement is performed in a fixed basis ([PauliZ, PauliI]) so that the circuit remains static.
    operation EvaluateCircuit(parameters : Double[], shots : Int) : Int {
        mutable totalZeroCount = 0;
        for _ in 1..shots {
            use qs = Qubit[2];
            PrepareAnsatz(qs, parameters);
            // Measure qubit 0 in the PauliZ basis (and ignore qubit 1)
            if (Measure([PauliZ, PauliI], qs) == Zero) {
                set totalZeroCount += 1;
            }
            ResetAll(qs);
        }
        return totalZeroCount;
    }

    /// # Summary
    /// Entry point for the VQE example that updates the parameters dynamically in a loop.
    @EntryPoint()
    operation Main() : Unit {
        mutable parameters = [0.222, 0.444];
        let shots = 1;
        let iterations = 5;
        for i in 1..iterations {
            let zeroCount = EvaluateCircuit(parameters, shots);
            // Update parameters by adding 0.01 to each element.
            set parameters w/= 0 <- parameters[0] + 0.111;
            set parameters w/= 1 <- parameters[1] + 0.111;
        }
    }
}
