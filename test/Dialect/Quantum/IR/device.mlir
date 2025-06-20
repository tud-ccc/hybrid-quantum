// RUN: quantum-opt %s -split-input-file

%dev = "quantum.device"() <{ coupling_graph = #quantum.coupling_graph<3, [[0, 1], [1, 2]]> }> : () -> !quantum.device<3, [[0, 1], [1, 2]]>

"quantum.circuit" () <{function_type = (!quantum.device<?, ?>) -> (i1), sym_name = "test_circuit"}>({
    ^bb0(%device : !quantum.device<3, [[0, 1], [1, 2]]>):
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    %m, %qm = "quantum.measure"(%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "quantum.deallocate"(%qm) : (!quantum.qubit<1>) -> ()
    "quantum.return"(%m) : (i1) -> ()
}) : () -> ()

%res = "quantum.instantiate"(%dev) <{circuit = @test_circuit}> : (!quantum.device<3, [[0, 1], [1, 2]]>) -> (i1)
