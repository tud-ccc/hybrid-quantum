// RUN: quantum-opt %s -split-input-file | FileCheck %s

// CHECK:   %[[DEVICE:.*]] = "qir.allocate_device"() <{coupling_graph = {{\[}}[0, 1], [1, 2]{{\]}}, num_qubits = 3 : i64}> : () -> !qir.device<3, {{\[}}[0, 1], [1, 2]{{\]}}>
%device = "qir.allocate_device"() <{coupling_graph = [[0, 1], [1, 2]], num_qubits = 3 : i64}> : () -> !qir.device<3, [[0, 1], [1, 2]]>

"qir.circuit"(%device) ({
  // Empty circuit body
}) : (!qir.device<3, [[0, 1], [1, 2]]>) -> ()


// CHECK:   "qir.circuit"(%[[DEVICE]]) ({
// CHECK:   }) : (!qir.device<3, {{\[}}[0, 1], [1, 2]{{\]}}>) -> ()
