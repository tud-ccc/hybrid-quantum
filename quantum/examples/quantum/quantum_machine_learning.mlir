//legacy code for quantum machine learning
//need to update nqubit types to qubit<n> type

module{

func.func @main() {
// Create tensors with float values
%num_qubits = arith.constant 2 : i32
%f0 = arith.constant 0.1 : f32
%f1 = arith.constant 0.2 : f32

//Training loop variables
%learning_rate = arith.constant 0.4 : f32
%shift         = arith.constant 0.3 : f32
%initial_theta  = tensor.from_elements %f1, %f0 : tensor<2xf32>
// %target        = tensor.from_elements %f0,%f0,%f0,%f0,%f1,%f1,%f1,%f1 : tensor<8xf32>
%singleTarget  = arith.constant 1.0: f32
%num_epochs    = arith.constant 100 : index

//Create a tensor of qubits
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index

%q0 = quantum.allocate(%num_qubits) : !quantum.nqubit
%q1 = quantum.extract %q0[0] : !quantum.nqubit -> !quantum.qubit
%q2 = quantum.extract %q0[1] : !quantum.nqubit -> !quantum.qubit
%qtensor_size = quantum.dim %q0, %num_qubits : !quantum.nqubit

//Create observables
//Define Hamiltonian, Cuda.spin.z equivalent
%PauliZ         = arith.constant 2 : i32
%PauliZobs      = quantum.namedobs %q1[%PauliZ] : !quantum.obs
%hamiltonian    = quantum.hamiltonian(%initial_theta : tensor<2xf32>) %PauliZobs,%PauliZobs : !quantum.obs

%current_theta_vals_X = tensor.extract %initial_theta[%c0] : tensor<2xf32>
%current_theta_vals_Y = tensor.extract %initial_theta[%c1] : tensor<2xf32>
%current_theta = tensor.from_elements %current_theta_vals_X, %current_theta_vals_Y : tensor<2xf32>

scf.for %i = %c0 to %num_epochs step %c1{
    //takes kernel+theta, returns a tensor<1x32> expectation value in this case
    %expVal                 = func.call @run(%current_theta, %q0, %hamiltonian)  : ( tensor<2xf32>, !quantum.nqubit, !quantum.obs) -> f32
    %loss, %grad_output     = func.call @lossfunc(%expVal, %singleTarget)  : (f32, f32) -> (f32, f32)
    %grad_val               = func.call @backward(%current_theta, %q0, %shift, %hamiltonian)   : (tensor<2xf32>, !quantum.nqubit, f32, !quantum.obs) -> f32
    %final_grad             = arith.mulf %grad_val, %grad_output: f32
    
    //Gradient update (using gradient descent)
    %new_theta_vals_sum = arith.mulf %learning_rate, %final_grad : f32
    %new_theta_vals_X   = arith.addf %new_theta_vals_sum, %current_theta_vals_X : f32

    //update theta values
    tensor.insert %new_theta_vals_X into %current_theta[%c0] : tensor<2xf32>
}
return
}

//Quantum Program. This part runs in the QPU.
func.func @kernel(%qvec: !quantum.nqubit, %angle0: f32, %angle1: f32, %hh : !quantum.obs) -> f32 {
        %a0 = arith.constant 0 : i32
        %a1 = arith.constant 1 : i32
        %qubit1 = quantum.extract %qvec[%a0]: !quantum.nqubit -> !quantum.qubit
        %qubit2 = quantum.extract %qvec[%a1]: !quantum.nqubit -> !quantum.qubit
        %rotated1 = quantum.R(%qubit1, x, %angle0)
        %rotated2 = quantum.R(%qubit2, y, %angle1)
        %exp = quantum.expectation %hh: f32  //what is the precise mappting to qir (possiblz map_to_z_basis function?)
        return %exp : f32
}

//run
func.func @run(%theta_vals: tensor<2xf32>, %qvector: !quantum.nqubit, %hh: !quantum.obs) -> f32 {
     // Get the size of the input tensor
     // Define constants for tensor indexing
    %a0 = arith.constant 0 : index
    %a1 = arith.constant 1 : index
    %theta1 = tensor.extract %theta_vals[%a0]: tensor<2xf32>
    %theta2 = tensor.extract %theta_vals[%a1]: tensor<2xf32>
    //change below to circuit not nqubit
    %exp_val = func.call @kernel(%qvector, %theta1, %theta2, %hh) : (!quantum.nqubit, f32, f32, !quantum.obs) -> f32
    return %exp_val : f32
}

func.func @lossfunc(%expVal: f32, %targetOutput : f32)-> (f32, f32){
    //Define a loss function here. Compute loss
    %loss = arith.subf %expVal, %targetOutput : f32
    %lossout = arith.mulf %loss, %loss  : f32
    %c2 = arith.constant 2.0 : f32
    %gradout = arith.mulf %loss, %c2 : f32
    return %lossout, %gradout : f32, f32
}


func.func @compute_gradient(%exp_vals_plus: f32, %exp_vals_minus: f32, %shift: f32)  -> f32 {
    %elemsum = arith.subf %exp_vals_plus, %exp_vals_minus: f32
    %c00 = arith.constant 0.5: f32
    %multiple = arith.mulf %shift, %c00 : f32
    %output = arith.mulf %elemsum, %multiple : f32
    return %output: f32
}

//Backward
func.func @backward(%thetas: tensor<2xf32>, %qvector: !quantum.nqubit, %shift: f32, %hh: !quantum.obs) -> f32 {
    // Define constants for indexing
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    //Unroll
    %theta0 = tensor.extract %thetas[%c0] : tensor<2xf32>
    %theta1 = tensor.extract %thetas[%c1] : tensor<2xf32>

    // Perturb the parameter
    %plus_val_0 = arith.addf %theta0, %shift : f32
    %minus_val_0 = arith.subf %theta0, %shift : f32
   
    //combine
    %thetas_plus = tensor.from_elements %plus_val_0, %theta1 : tensor<2xf32>
    %thetas_minus = tensor.from_elements %minus_val_0, %theta1 : tensor<2xf32>

    // Run quantum circuit with perturbed parameters (plus)
    %exp_vals_plus = func.call @run(%thetas_plus, %qvector, %hh) : (tensor<2xf32>, !quantum.nqubit, !quantum.obs) -> f32
    %exp_vals_minus = func.call @run(%thetas_minus, %qvector, %hh) : (tensor<2xf32>, !quantum.nqubit, !quantum.obs) -> f32
    %out_grad = func.call @compute_gradient(%exp_vals_plus, %exp_vals_minus, %shift): (f32, f32, f32) -> f32
    return %out_grad : f32
}
}