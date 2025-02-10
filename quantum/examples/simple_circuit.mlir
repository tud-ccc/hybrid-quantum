module {
  func.func @main() {
    %q1 = quantum.alloc : !quantum.qubit<10>
    quantum.H %q1: !quantum.qubit<10>
    return
  }
}
