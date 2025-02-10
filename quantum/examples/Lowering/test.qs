import Microsoft.Quantum.Diagnostics.*;

@EntryPoint()
operation SimpleQubitOperation() : Result {
    use q = Qubit();
    H(q);
    let result = M(q);
    Reset(q);
    return result;
}
