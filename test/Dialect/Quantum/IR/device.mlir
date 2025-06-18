// RUN: quantum-opt %s -split-input-file

%dev = "quantum.device"() { coupling_graph = #quantum.coupling_graph<3, [[0, 1], [1, 2]]> } : () -> !quantum.device<3, [[0, 1], [1, 2]]>
