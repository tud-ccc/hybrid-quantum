// RUN: quantum-opt %s -verify-diagnostics

module {

    func.func @qubit_multiple_uses_same_region() -> () {
        // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %q2 = "quantum.Z" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.deallocate" (%q1) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q2) : (!quantum.qubit<1>) -> ()
        return 
    }

    func.func @qubit_multiple_uses_different_region(%b : i1) -> () {
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // expected-error@+1 {{'quantum.if' op captured qubit #0 used more than once within the same block}}
        %r = quantum.if %b qubits(%qin = %q) -> (!quantum.qubit<1>) {
            %q1 = "quantum.X" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            "quantum.yield" (%q1) : (!quantum.qubit<1>) -> ()
        } else {
            %q2 = "quantum.H" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            %q3 = "quantum.Y" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            "quantum.yield" (%q2) : (!quantum.qubit<1>) -> ()
        }
        "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
        return 
    }
}