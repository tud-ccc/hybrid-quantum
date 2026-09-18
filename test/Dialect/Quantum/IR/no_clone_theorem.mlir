//RUN: quantum-opt %s -split-input-file -verify-linearity -verify-diagnostics

func.func @qubit_multiple_uses_same_region() -> () {
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // expected-error@+1 {{qubit operand #0 has already been consumed}}
    %q2 = "quantum.Z" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    "quantum.deallocate" (%q1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%q2) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

// CHECK-LABEL: @qubit_multiple_uses_else_region_positive
func.func @qubit_multiple_uses_else_region_positive(%b : i1) -> () {
    %q = "quantum.alloc" () : () -> !quantum.qubit<1>
    %r = scf.if %b -> (!quantum.qubit<1>) {
        %qX = "quantum.X" (%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %qX : !quantum.qubit<1>
    } else {
        %qY = "quantum.Y" (%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %qY : !quantum.qubit<1>
    }
    "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
    return
}

 // -----

func.func @qubit_multiple_uses_else_region(%b : i1) -> () {
    %q = "quantum.alloc" () : () -> !quantum.qubit<1>
    %r = scf.if %b -> (!quantum.qubit<1>) {
        %qX = "quantum.X" (%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %qX : !quantum.qubit<1>
    } else {
        %qH = "quantum.H" (%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        // expected-error@+1 {{qubit operand #0 has already been consumed}}
        %qY = "quantum.Y" (%q) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %qY : !quantum.qubit<1>
    }
    "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
    return
}

 // -----

func.func @qubit_multiple_uses_then_region(%b : i1) -> () {
    %q1 = "quantum.alloc" () : () -> !quantum.qubit<1>
    %q2 = "quantum.alloc" () : () -> !quantum.qubit<1>

    %r1, %r2 = scf.if %b -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        %qH = "quantum.H" (%q2) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        // expected-error@+1 {{qubit operand #0 has already been consumed}}
        %qY = "quantum.Y" (%q2) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %q1, %qY : !quantum.qubit<1>, !quantum.qubit<1>
    } else {
        %qX = "quantum.X" (%q1) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        scf.yield %qX, %q2 : !quantum.qubit<1>, !quantum.qubit<1>
    }
    "quantum.deallocate" (%r1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%r2) : (!quantum.qubit<1>) -> ()
    return
}

 // -----
