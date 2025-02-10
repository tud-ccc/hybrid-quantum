    // operation CreateBellPair() : Unit {
    //     use qubit1 = Qubit();
    //     use qubit2 = Qubit();
    //     H(qubit1);
    //     CNOT(qubit1, qubit2);
    //     // Measure the qubits (optional)
    //     let result1 = M(qubit1);
    //     let result2 = M(qubit2);
    // }

//The Q# code above is turned into implementation in our dialect
module {
    func.func @entrypt() -> (i1, i1) {
        //Declare array. Extract each qubit
        %qubit1 = quantum.alloc : !quantum.qubit<1>

        // Measure qubits
        %result1, %collapsedqubit1 = quantum.measure %qubit1 : i1, !quantum.qubit<1>
        
        return %result1 : i1  
    }
}
