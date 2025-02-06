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
        %qubits = quantum.alloc : !quantum.qubit<2>
        %qubit:2 = quantum.extract %qubits[0,1]: !quantum.qubit<2> -> !quantum.qubit<1>

        //Apply H and CNOT gate simultaneously
        %applyH = quantum.H %qubit1 : !quantum.qubit
        %applyCNOT:2 = quantum.CNOT %qubit1, %qubit2
        
        // Measure qubits
        %result1, %collapsedqubit1 = quantum.measure %qubit1 : i1, !quantum.qubit
        %result2, %collapsedqubit2 = quantum.measure %qubit2 : i1, !quantum.qubit
        
        return %result1, %result2 : i1, i1  
    }
}
