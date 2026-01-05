//RUN: quantum-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

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
    %pred = rvsdg.match(%b : i1) [
        #rvsdg.matchRule<0 -> 1>,
        #rvsdg.matchRule<1 -> 0>
    ] -> !rvsdg.ctrl<2>
    %r = rvsdg.gamma (%pred : !rvsdg.ctrl<2>) (%q : !quantum.qubit<1>):[
    (%qin: !quantum.qubit<1>):{
        %qX = "quantum.X" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qX : !quantum.qubit<1>)
    },
    (%qin: !quantum.qubit<1>):{
        // expected-error@+1 {{'quantum.H' op qubit #0 is used 2 times}}
        %qH = "quantum.H" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %qY = "quantum.Y" (%qin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qY : !quantum.qubit<1>)
    }
    ] -> !quantum.qubit<1>
    "quantum.deallocate" (%r) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

 func.func @qubit_multiple_uses_then_region(%b : i1) -> () {
    %q1 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %q2 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %pred = rvsdg.match(%b : i1) [
        #rvsdg.matchRule<0 -> 1>,
        #rvsdg.matchRule<1 -> 0>
    ] -> !rvsdg.ctrl<2>
    %r1, %r2 = rvsdg.gamma (%pred : !rvsdg.ctrl<2>) (%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>):[
    (%qin1: !quantum.qubit<1>, %qin2: !quantum.qubit<1>):{
        // expected-error@+1 {{'quantum.H' op qubit #0 is used 2 times}}
        %qH = "quantum.H" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %qY = "quantum.Y" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qin1 : !quantum.qubit<1>, %qY: !quantum.qubit<1>)
    },
    (%qin1: !quantum.qubit<1>, %qin2: !quantum.qubit<1>):{
        %qX = "quantum.X" (%qin1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qX : !quantum.qubit<1>, %qin2 : !quantum.qubit<1>)
    }
    ] -> !quantum.qubit<1>, !quantum.qubit<1>
    "quantum.deallocate" (%r1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%r2) : (!quantum.qubit<1>) -> ()
    return 
}

 // -----

 func.func @qubit_multiple_uses_in_capture_clause(%b : i1) -> () {
    // expected-error@+1 {{'quantum.alloc' op result qubit #0 used more than once within the same block}}
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    %pred = rvsdg.match(%b : i1) [
        #rvsdg.matchRule<0 -> 1>,
        #rvsdg.matchRule<1 -> 0>
    ] -> !rvsdg.ctrl<2>
    %r1, %r2 = rvsdg.gamma (%pred : !rvsdg.ctrl<2>) (%q : !quantum.qubit<1>, %q : !quantum.qubit<1>):[
    (%qin1: !quantum.qubit<1>, %qin2: !quantum.qubit<1>):{
        %qY = "quantum.Y" (%qin2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qin1: !quantum.qubit<1>, %qY: !quantum.qubit<1>)
    },
    (%qin1: !quantum.qubit<1>, %qin2: !quantum.qubit<1>):{
        %qX = "quantum.X" (%qin1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        rvsdg.yield (%qX: !quantum.qubit<1>, %qin2: !quantum.qubit<1>)
    }
    ] -> !quantum.qubit<1>,!quantum.qubit<1>
    "quantum.deallocate" (%r1) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%r2) : (!quantum.qubit<1>) -> ()
    return 
}
