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
        // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        %q5 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %q6 = "quantum.Z" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %l, %r = scf.if %b -> (!quantum.qubit<1>, !quantum.qubit<1>) {
            // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
            %q1 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
            //%q1 = "quantum.X" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            scf.yield %q1, %q1 : !quantum.qubit<1>, !quantum.qubit<1>
        } else {
            %q2 = "quantum.Y" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            %q3 = "quantum.Y" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            scf.yield %q2, %q3 : !quantum.qubit<1>, !quantum.qubit<1>
        }
        "quantum.deallocate" (%l) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
        return 
    }
}