// RUN: quantum-opt --quantum-optimise %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @main(
  func.func @main() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q2 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q3 = "quantum.X" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    %q4 = "quantum.X" (%q3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: return %[[Q]] : !quantum.qubit<1>
    return %q4 : !quantum.qubit<1>
  }
}

