//TODO: 
//Add quantum ansatz
//Add measure energy
// 

// Define the quantum circuit with the initial parameters set across n qubit system. 
func.func @QuantumAnsatz(%parameters: tensor<3xf32>, %number_of_qubits: i64) -> !quantum.nqubit{

    %qarray = quantum.allocate(%number_of_qubits) : !quantum.nqubit    
    %qoutput = quantum.allocate(2) : !quantum.nqubit
    %qubit1 = quantum.extract %q0[0] : !quantum.nqubit -> !quantum.qubit
    %qubit2 = quantum.extract %q0[1] : !quantum.nqubit -> !quantum.qubit

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %param0 = tensor.extract %variationalParams[%c0] : tensor<3xf32>
    %param1 = tensor.extract %variationalParams[%c1] : tensor<3xf32>
    %param2 = tensor.extract %variationalParams[%c2] : tensor<3xf32>
  
    %q1 = quantum.U %qubit1(2.00, 0.10, 1.20) 
    %q2 = quantum.U %qubit2(0.00, 1.07, 2.03) 
    %q3, %q4 = quantum.CNOT %q1,%q2
    %q1_result, %newq1 = quantum.measure %q3
    %q2_result, %newq2 = quantum.measure %q4

    // Store the measured qubits in the new array
    quantum.insert %qoutput[0], %newq1 : !quantum.nqubit, !quantum.qubit
    quantum.insert %qoutput[1], %newq2 : !quantum.nqubit, !quantum.qubit
    return %qoutput : !quantum.nqubit
}

  // Define a function for measuring energy based on measurement results
  func.func @MeasureEnergy(%results: !quantum.nqubit) -> f32 {
    // Here you can implement logic to calculate energy based on measurement results
    // For simplicity, let's assume we return a dummy energy value.
    %constout = arith.constant 0.0: f32
    return %constant : f32
  }

  // Define a function for the Variational Quantum Eigensolver
  func.func @QuantumFunction(%updatedParams: tensor<3xf32>) -> f32 {
    // llvm.call Quantum Ansatz
    %numQubits = arith.constant 2: i64
    %qoutput = llvm.call @QuantumAnsatz(%updatedParams, %numQubits) : (tensor<3xf32>, i64) -> !quantum.nqubit
    
    // Measure energy based on output from Quantum Ansatz
    return llvm.call @MeasureEnergy(%qoutput) : (!quantum.nqubit) -> f32
  }

// Define a function for the forward pass that returns a tensor.
func.func @forward(%params: tensor<2xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tmep_iter = arith.constant 1000.0 : f32
    %iterations = arith.constant 1000 : index
    %sum = arith.constant 0.0 : f32

    %result = scf.for %i = %c0 to %iterations step %c1 iter_args(%current_sum = %sum) -> (f32) {
        %measurement_float = llvm.call @QuantumFunction(%params) : (tensor<2xf32>) -> f32
        //%measurement_float = arith.uitofp %measurement : i1 to f32
        %new_sum = arith.addf %current_sum, %measurement_float : f32
        scf.yield %new_sum : f32
    }

    %avg = arith.divf %result, %tmep_iter : f32
    return %avg : f32
}

// Define a function for the backward pass that returns a tensor.
func.func @backward(%params: tensor<2xf32>, %epsilon: f32) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %two = arith.constant 2.0 : f32

    %base_loss = llvm.call @forward(%params) : (tensor<2xf32>) -> f32

    %gradients = scf.for %i = %c0 to %c2 step %c1 iter_args(%current_gradients = %params) -> (tensor<2xf32>) {
        %param_i = tensor.extract %params[%i] : tensor<2xf32>
        %perturbed_param = arith.addf %param_i, %epsilon : f32
        %perturbed_params = tensor.insert %perturbed_param into %params[%i] : tensor<2xf32>
        
        %perturbed_loss = llvm.call @forward(%perturbed_params) : (tensor<2xf32>) -> f32
        %loss_diff = arith.subf %perturbed_loss, %base_loss : f32
        %gradient = arith.divf %loss_diff, %epsilon : f32
        
        %new_gradients = tensor.insert %gradient into %current_gradients[%i] : tensor<2xf32>
        scf.yield %new_gradients : tensor<2xf32>
    }

    return %gradients : tensor<2xf32>
}

// Define a function to run the quantum circuit and optimize parameters
func.func @optimise() -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max_iterations = arith.constant 100 : index
    %learning_rate = arith.constant 0.01 : f32
    %epsilon = arith.constant 0.001 : f32

    %initial_params = tensor.from_elements %learning_rate, %learning_rate : tensor<2xf32>

    %final_params = scf.for %i = %c0 to %max_iterations step %c1 iter_args(%current_params = %initial_params) -> (tensor<2xf32>) {
        %loss = llvm.call @forward(%current_params) : (tensor<2xf32>) -> f32
        %gradients = llvm.call @backward(%current_params, %epsilon) : (tensor<2xf32>, f32) -> tensor<2xf32>

        %updated_params = scf.for %j = %c0 to %c2 step %c1 iter_args(%params = %current_params) -> (tensor<2xf32>) {
            %param_j = tensor.extract %params[%j] : tensor<2xf32>
            %gradient_j = tensor.extract %gradients[%j] : tensor<2xf32>
            %update = arith.mulf %learning_rate, %gradient_j : f32
            %new_param = arith.subf %param_j, %update : f32
            %new_params = tensor.insert %new_param into %params[%j] : tensor<2xf32>
            scf.yield %new_params : tensor<2xf32>
        }

        scf.yield %updated_params : tensor<2xf32>
    }

    return %final_params : tensor<2xf32>
}


// Main function to orchestrate execution
func.func @main() -> tensor<3xf32> {
    // Define initial parameters for the quantum function
    %set_number_of_qubits = arith.constant 2: i64
    %t = arith.constant 0.0 : f32
    %initial_params = tensor.from_elements %t, %t, %t : tensor<3xf32>
    
    // Call QuantumFunction to get an initial energy estimate
    %energy_estimate = call @QuantumFunction(%initial_params) : (tensor<3xf32>) -> f32
    
    // Print or log the initial energy estimate (optional)
    
    // Now call optimise to refine parameters based on energy measurements
    %optimized_params = call @optimise() : () -> tensor<3xf32>
    
    return %optimized_params : tensor<3xf32>
}