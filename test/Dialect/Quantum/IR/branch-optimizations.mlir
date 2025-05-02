// RUN: quantum-opt %s -cse -loop-invariant-code-motion -control-flow-sink | FileCheck %s

module {
    
    // CHECK-LABEL: func.func @hoist_op_from_if(
    func.func @hoist_op_from_if(%b : i1) -> () {
        %q = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-DAG: "quantum.split"
        %a1, %b1 = "quantum.split" (%q) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
        // CHECK-NEXT: "quantum.H"
        // CHECK-NEXT: quantum.if
        %aout, %bout = quantum.if %b qubits(%ain = %a1, %bin = %b1) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
            %b2 = "quantum.X" (%bin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            // CHECK-NOT "quantum.H"
            %a2 = "quantum.H" (%ain) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            %b3 = "quantum.X" (%b2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            "quantum.yield" (%a2, %b3) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
        } else {
            %b2 = "quantum.Z" (%bin) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            // CHECK-NOT "quantum.H"
            %a2 = "quantum.H" (%ain) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            %b3 = "quantum.Z" (%b2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
            "quantum.yield" (%a2, %b3) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
        }
        %out = "quantum.merge" (%aout, %bout) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)
        "quantum.deallocate" (%out) : (!quantum.qubit<2>) -> ()
        return
    }
}