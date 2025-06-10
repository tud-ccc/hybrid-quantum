// RUN: quantum-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @qubit_multiple_uses_same_region() -> () {
    // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q2 = "quantum.Z" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    "quantum.deallocate" (%q1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%q2) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

func.func @qubit_multiple_uses_else_region(%b : i1) -> () {
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    // expected-error@+1 {{'quantum.if' op captured qubit #0 used more than once within the same block}}
    %r = quantum.if %b ins(%qin = %q) -> (!quantum.qubit<1>) {
        %qX = "quantum.X" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qX) : (!quantum.qubit<1>) -> ()
    } else {
        %qH = "quantum.H" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %qY = "quantum.Y" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qY) : (!quantum.qubit<1>) -> ()
    }
    "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

 func.func @qubit_multiple_uses_then_region(%b : i1) -> () {
    %q1 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %q2 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    // expected-error@+1 {{'quantum.if' op captured qubit #1 used more than once within the same block}}
    %r1, %r2 = quantum.if %b ins(%qin1 = %q1, %qin2 = %q2) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        %qH = "quantum.H" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %qY = "quantum.Y" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qin1, %qY) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    } else {
        %qX = "quantum.X" (%qin1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qX, %qin2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    }
    "quantum.deallocate" (%r1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%r2) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

 func.func @qubit_multiple_uses_in_capture_clause(%b : i1) -> () {
    // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %r1, %r2 = quantum.if %b ins(%qin1 = %q, %qin2 = %q) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        %qY = "quantum.Y" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qin1, %qY) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    } else {
        %qX = "quantum.X" (%qin1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.yield" (%qX, %qin2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    }
    "quantum.deallocate" (%r1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%r2) : (!quantum.qubit<1>) -> ()
    return 
}