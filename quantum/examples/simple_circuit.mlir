module {
  func.func @test_quantum_ops() {
    %c1 = arith.constant 5 : i32
    %c2 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    
    //allocate a tensor
    %reg = quantum.alloc : !quantum.qubit<10>
    return
  }
}