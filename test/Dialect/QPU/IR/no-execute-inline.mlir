// RUN: quantum-opt %s -split-input-file -inline

qpu.module @test {
    // CHECK-SAME
    "qpu.circuit"() <{function_type = () -> (i1), sym_name = "test_circuit"}>({
        ^bb0():
        %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
        %m, %qm = "quantum.measure_single"(%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        "quantum.deallocate"(%qm) : (!quantum.qubit<1>) -> ()
        "qpu.return"(%m) : (i1) -> ()
    }) : () -> ()
}

func.func @main() -> i1 {
    %res = arith.constant false
    // CHECK-SAME
    qpu.execute @test::@test_circuit args() outs(%res : i1)
    func.return %res : i1
}
