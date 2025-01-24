// RUN: quantum-opt %s | FileCheck %s

// CHECK: alloc
%reg = quantum.alloc : !quantum.qubit<3>