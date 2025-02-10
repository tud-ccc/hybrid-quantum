module {
  func.func @main() {
    %qubit_array = quantum.alloc : !quantum.qubit {ids ={0, 1, 2}} <3>
    %qubit1, %qubit2 = quantum.split %qubit_array : !quantum.qubit {ids ={0, 1, 2}} <3> -> !quantum.qubit {ids ={0}} <1>, !quantum.qubit {ids ={1, 2}} <2>
    %qubit1_2 = quantum.H %qubit1 : !quantum.qubit {id ={2}} <1>
    return 
  }
}