// RUN: quantum-opt %s -verify-diagnostics

module {
    
    func.func @qubit_used_multiple_times() -> () {
        // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %q2 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        return 
    }
}