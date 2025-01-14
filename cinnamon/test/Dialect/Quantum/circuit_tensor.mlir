
// Create tensors with float values
%num_qubits = arith.constant 4 : i32
%f0 = arith.constant 0.1 : f32
%f1 = arith.constant 0.2 : f32
%init_params = tensor.from_elements %f0, %f1 : tensor<2xf32>
%initial_params = tensor.cast %init_params : tensor<2xf32> to tensor<?xf32>
%init_grad = tensor.from_elements %f1, %f0 : tensor<2xf32>
%initial_grad = tensor.cast %init_grad : tensor<2xf32> to tensor<?xf32>

//Create a tensor of qubits
%q0 = quantum.allocate(%num_qubits) : !quantum.nqubit
%q1 = quantum.extract %q0[0] : !quantum.nqubit -> !quantum.qubit
%q2 = quantum.extract %q0[1] : !quantum.nqubit -> !quantum.qubit
%q3 = tensor.from_elements %q1, %q2 : tensor<2x!quantum.qubit>
%q3_dynamic = tensor.cast %q3 : tensor<2x!quantum.qubit> to tensor<?x!quantum.qubit>

//Create observables
//Define Hamiltonian, Cuda.spin.z equivalent
%PauliZ         = arith.constant 2 : i32
%PauliZobs      = quantum.namedobs %q1[%PauliZ] : !quantum.obs
%hamiltonian    = quantum.hamiltonian(%init_params : tensor<2xf32>) %PauliZobs,%PauliZobs : !quantum.obs

//test call all 4 functions to check functionality
%test_run       = func.call @run(%initial_params, %q3_dynamic, %hamiltonian)  : ( tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs) -> tensor<?xf32>
%test_forward   = func.call @forward(%initial_params, %q3_dynamic, %hamiltonian, %f1)  : ( tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs, f32) -> tensor<?xf32>
%test_backward  = func.call @backward(%initial_params, %q3_dynamic, %hamiltonian, %f1, %initial_grad)  : ( tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs, f32, tensor<?xf32>) -> tensor<?xf32>

//hoist all other dialects before calling the kernel

%test_qpu       = quantum.call_circuit @kernel(%q3_dynamic, %num_qubits, %initial_params) : (tensor<?x!quantum.qubit>, i32, tensor<?xf32>) -> tensor<?x!quantum.qubit>


//Quantum Program. This part runs in the QPU.
quantum.circuit @kernel(%qvector: tensor<?x!quantum.qubit>, %nn: i32, %angles: tensor<?xf32>) -> tensor<?x!quantum.qubit> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index  
    %tensor_size = tensor.dim %qvector, %c0 : tensor<?x!quantum.qubit>
    %empty_tensor = tensor.empty(%tensor_size) : tensor<?x!quantum.qubit>
    %angless = tensor.cast %angles: tensor<?xf32> to tensor<2xf32>
    %angle0 = tensor.extract %angless[%c0] : tensor<2xf32>
    %angle1 = tensor.extract %angless[%c1] : tensor<2xf32>

    %result = scf.for %i = %c0 to %tensor_size step %c1 
                  iter_args(%current_tensor = %empty_tensor) -> (tensor<?x!quantum.qubit>) {
                  %nqubit = tensor.extract %qvector[%i] : tensor<?x!quantum.qubit>
                  %rotated = quantum.R(%nqubit, x, %angle0)
                  %inserted = tensor.insert %rotated into %current_tensor[%i] : tensor<?x!quantum.qubit>
              scf.yield %inserted : tensor<?x!quantum.qubit>
              }
    quantum.return %result : tensor<?x!quantum.qubit>
}

//run
func.func @run(%theta_vals: tensor<?xf32>, %qvector: tensor<?x!quantum.qubit>, %hh: !quantum.obs) -> tensor<?xf32> {
     // Get the size of the input tensor
     // Define constants for tensor indexing
    %a0 = arith.constant 0 : index
    %a1 = arith.constant 1 : index
    %size = tensor.dim %theta_vals, %a0 : tensor<?xf32>
    %empty_result = tensor.empty(%size) : tensor<?xf32>
    
    // Loop over the input tensor
    %result = scf.for %i = %a0 to %size step %a1 
        iter_args(%current_result = %empty_result) -> (tensor<?xf32>) {
        %theta = tensor.extract %theta_vals[%i] : tensor<?xf32>   
        %obs_result = quantum.observe %qvector,%hh, %theta_vals : tensor<?x!quantum.qubit>, tensor<?xf32> -> !quantum.obs
        %exp_val = quantum.expectation %obs_result : f32
        %new_result = tensor.insert %exp_val into %current_result[%i] : tensor<?xf32>
        scf.yield %new_result : tensor<?xf32>
    }
    return %result : tensor<?xf32>
}

// Forward function
func.func @forward(%theta_vals: tensor<?xf32>, %qvector: tensor<?x!quantum.qubit>, %hh: !quantum.obs, %shift: f32) -> tensor<?xf32> {
    // Define a new tensor to hold the copied values of qvector
    %a0 = arith.constant 0 : index
    %a1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %size = tensor.dim %theta_vals, %a0 : tensor<?xf32>
    %cqvector = tensor.empty(%size) : tensor<?x!quantum.qubit>
    tensor.insert_slice %qvector into %cqvector[%a0][%size][%a1] : tensor<?x!quantum.qubit> into tensor<?x!quantum.qubit>
    %cshift = arith.addf %shift, %zero : f32

    // Perform further operations as needed...
    %result = func.call @run(%theta_vals, %qvector, %hh) : (tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs) ->  tensor<?xf32>
    
    // Placeholder for the result. Replace with actual computation.
    return %result : tensor<?xf32>
}

func.func @compute_gradient(%exp_vals_plus: tensor<?xf32>, %exp_vals_minus: tensor<?xf32>, %shift: f32)  -> f32 {
    %output = arith.constant 0.0: f32
    return %output: f32
}

//Backward
func.func @backward(%thetas: tensor<?xf32>, %quantum_circuit: tensor<?x!quantum.qubit>, %hh: !quantum.obs, %shift: f32, %grad_output: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %size = tensor.dim %thetas, %c0 : tensor<?xf32>
    %gradients = tensor.empty(%size) : tensor<?xf32>

    %gradients_result = scf.for %i = %c0 to %size step %c1 iter_args(%current_gradients = %gradients) -> (tensor<?xf32>) {
        // Create copies of the original thetas for perturbation
        %thetas_plus = tensor.empty(%size) : tensor<?xf32>
        %thetas_minus = tensor.empty(%size) : tensor<?xf32>
        %thetas_plus_filled = tensor.insert_slice %thetas into %thetas_plus[%c0][%size][%c1] : tensor<?xf32> into tensor<?xf32>
        %thetas_minus_filled = tensor.insert_slice %thetas into %thetas_minus[%c0][%size][%c1] : tensor<?xf32> into tensor<?xf32>

        // Extract and update the i-th parameter
        %theta_val = tensor.extract %thetas[%i] : tensor<?xf32>
        %plus_val = arith.addf %theta_val, %shift : f32
        %minus_val = arith.subf %theta_val, %shift : f32

        // Update plus and minus tensors
        %thetas_plus_updated = tensor.insert %plus_val into %thetas_plus_filled[%i] : tensor<?xf32>
        %thetas_minus_updated = tensor.insert %minus_val into %thetas_minus_filled[%i] : tensor<?xf32>

        // Run quantum circuit with perturbed parameters
        %exp_vals_plus = func.call @run(%thetas_plus_updated, %quantum_circuit, %hh) : (tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs) -> tensor<?xf32>
        %exp_vals_minus = func.call @run(%thetas_minus_updated, %quantum_circuit, %hh) : (tensor<?xf32>, tensor<?x!quantum.qubit>, !quantum.obs) -> tensor<?xf32>
        %gradient_val = func.call @compute_gradient(%exp_vals_plus, %exp_vals_minus, %shift) : (tensor<?xf32>, tensor<?xf32>, f32) -> f32
        %updated_gradients = tensor.insert %gradient_val into %current_gradients[%i] : tensor<?xf32>
        scf.yield %updated_gradients : tensor<?xf32>
    }

    %final_gradients = arith.mulf %gradients_result, %grad_output : tensor<?xf32>
    return %final_gradients : tensor<?xf32>
}