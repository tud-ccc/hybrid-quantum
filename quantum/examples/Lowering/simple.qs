import Microsoft.Quantum.Diagnostics.*;

operation Main() : Result {  

    use q1 = Qubit();
    
    H(q1);
    
    let m1 = M(q1);
    
    Reset(q1);

    return m1;
}