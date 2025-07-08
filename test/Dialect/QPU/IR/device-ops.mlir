// RUN: quantum-opt %s -split-input-file

qpu.module @test
    [#qpu.target<
        qubits = 3,
        coupling_graph = [[0, 1], [1, 2]]
    >] {

    "qpu.circuit"() <{function_type = (f64) -> (i1), sym_name = "test_circuit"}>({
        ^bb0(%theta : f64):
        %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
        %m, %qm = "quantum.measure_single"(%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        "quantum.deallocate"(%qm) : (!quantum.qubit<1>) -> ()
        "qpu.return"(%m) : (i1) -> ()
    }) : () -> ()
}

%theta = arith.constant 3.2 : f64
%out = "qpu.execute" @test::@test_circuit(%theta) : (f64) -> (i1)
