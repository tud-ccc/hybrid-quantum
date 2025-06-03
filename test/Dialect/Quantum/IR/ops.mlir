// RUN: quantum-opt %s -split-input-file

func.func @qubit_alloc_dealloc() -> () {
    %reg = "quantum.alloc" () : () -> (!quantum.qubit<3>)
    "quantum.deallocate" (%reg) : (!quantum.qubit<3>) -> ()
    return
}

// -----

func.func @qubit_split_merge(%reg : !quantum.qubit<3>) -> (!quantum.qubit<3>) {
    // split
    %q0, %q1 = "quantum.split" (%reg) : (!quantum.qubit<3>) -> (!quantum.qubit<2>, !quantum.qubit<1>)
    %q00, %q01 = "quantum.split" (%q0) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    // merge
    %qm0 = "quantum.merge" (%q00, %q01) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)
    %qm1 = "quantum.merge" (%qm0, %q1) : (!quantum.qubit<2>, !quantum.qubit<1>) -> (!quantum.qubit<3>)
    return %qm1 : !quantum.qubit<3>
}

// -----

func.func @qubit_measure_single(%reg : !quantum.qubit<1>) -> (i1, !quantum.qubit<1>) {
    %m, %qm_single_out = "quantum.measure_single" (%reg) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    return %m, %qm_single_out : i1, !quantum.qubit<1>
}

// -----

func.func @qubit_measure_multiple(%reg : !quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>) {
    %mt, %qm_out = "quantum.measure" (%reg) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
    return %mt, %qm_out : tensor<2xi1>, !quantum.qubit<2>
}

// -----

func.func @qubit_pauli(%reg : !quantum.qubit<1>) -> (!quantum.qubit<1>) {
    %qX = "quantum.X" (%reg) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %qY = "quantum.Y" (%qX) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %qZ = "quantum.Z" (%qY) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %qH = "quantum.H" (%qZ) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %qH : !quantum.qubit<1>
}

// -----

func.func @quantum_if(%b : i1, %reg : !quantum.qubit<1>) -> (!quantum.qubit<1>) {
    %q = quantum.if %b ins(%qin = %reg) -> (!quantum.qubit<1>) {
        "quantum.yield" (%qin) : (!quantum.qubit<1>) -> ()
    } else {
        "quantum.yield" (%qin) : (!quantum.qubit<1>) -> ()
    }
    return %q : !quantum.qubit<1>
}

// -----

"quantum.gate"() <{function_type = (!quantum.qubit<1>) -> (!quantum.qubit<1>), sym_name = "test"}>({
    ^bb0(%arg0 : !quantum.qubit<1>):
    "quantum.return"(%arg0) : (!quantum.qubit<1>) -> ()
}) : () -> ()

func.func @quantum_gate(%reg : !quantum.qubit<1>) -> (!quantum.qubit<1>) {
    %out = "quantum.call"(%reg) <{callee = @test}> : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %out : !quantum.qubit<1>
}
