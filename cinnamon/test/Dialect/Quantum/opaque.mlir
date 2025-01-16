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
        %qubits = quantum.init_array(2) : !quantum.nqubit
        %qubit1 = quantum.extract %qubits[0]: !quantum.nqubit -> !quantum.qubit
        %qubit2 = quantum.extract %qubits[1]: !quantum.nqubit -> !quantum.qubit
        %applyH = quantum.H %qubit1 : !quantum.qubit
        %applyCNOT:2 = quantum.CNOT %qubit1, %qubit2
        
        // Measure qubits
        %result1, %collapsedqubit1 = quantum.measure %qubit1 : i1, !quantum.qubit
        %result2, %collapsedqubit2 = quantum.measure %qubit2 : i1, !quantum.qubit
        return %result1, %result2 : i1, i1  
    }
}
