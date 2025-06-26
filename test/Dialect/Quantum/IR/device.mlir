// RUN: quantum-opt %s -split-input-file

%dev = "quantum.device"() <{ coupling_graph = #quantum.coupling_graph<3, [[0, 1], [1, 2]]> }> : () -> !quantum.device<3, [[0, 1], [1, 2]]>
%dev2 = "quantum.device"() <{ coupling_graph = #quantum.coupling_graph<3, [[0, 1], [1, 2]]> }> : () -> !quantum.device<3, [[0, 1], [1, 2]]>

%circ = "quantum.circuit"() <{circuit_type = (f64) -> (i1), name = "test_circuit"}>({
    ^bb0(%theta : f64):
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    %m, %qm = "quantum.measure_single"(%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "quantum.deallocate"(%qm) : (!quantum.qubit<1>) -> ()
    "quantum.return"(%m) : (i1) -> ()
}) : () -> (!quantum.circuit)

%phys_circ = "quantum.instantiate"(%dev, %circ) : (!quantum.device<3, [[0, 1], [1, 2]]>, !quantum.circuit) -> (!quantum.circuit)

%out = "quantum.execute"(%phys_circ) : (!quantum.circuit) -> (i1)
