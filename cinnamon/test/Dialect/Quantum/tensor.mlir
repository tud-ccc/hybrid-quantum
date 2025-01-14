module{
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

}