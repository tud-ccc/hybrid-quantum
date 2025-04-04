# Quantum MLIR Examples

This directory contains several MLIR examples demonstrating various quantum algorithms. These examples illustrate how quantum circuits and protocols can be represented using MLIR and the QIR dialect. The following examples are included:

- `bell_states.mlir`
- `simple_vqe.mlir`
- `superposition.mlir`
- `teleportation.mlir`

---

## Detailed Examples

### 1. Bell States (`bell_states.mlir`)
**Description:**  
This example implements the preparation and measurement of all four Bell states (or EPR pairs). Bell states are specific two-qubit quantum states that exhibit maximal entanglement. The MLIR code demonstrates how to generate and measure the four different Bell states using `scf.for` loop:

- **|Φ+⟩:** Prepared by applying a Hadamard gate to the first qubit followed by a CNOT gate.
- **|Φ-⟩:** Similar to |Φ+⟩, but with an additional Z gate on the first qubit to flip the phase.
- **|Ψ+⟩:** Created by applying a Hadamard gate on the first qubit, then applying an X gate on the second qubit before the CNOT.
- **|Ψ-⟩:** Prepared by applying a Hadamard gate on the first qubit, then a Z gate on the first qubit and an X gate on the second qubit before the CNOT.

**Process:**  
- The MLIR code uses an `scf.for` loop that iterates four times.
- In each iteration, two qubits are allocated.
- Based on the loop index, a conditional branch applies the corresponding sequence of quantum gates to prepare one of the four Bell states.
- Each pair of qubits is then measured. The measurement operations return 1D tensors, from which scalar classical bits are extracted.

---

### 2. Simple VQE (`simple_vqe.mlir`)
**Description:**  
This example implements a simplified version of the Variational Quantum Eigensolver (VQE) algorithm. VQE is a hybrid quantum-classical method used to estimate the ground state energy of quantum systems.
  
**Detailed Process:**
- **Initialization:**  
  Two theta parameters (angles) are stored in memory. These parameters govern the rotation angles in the quantum circuit.
  
- **Quantum Kernel Execution:**  
  The quantum kernel reads the theta values and applies parameterized rotations using `Rx` and `Rz` gates on the qubits. A Hadamard gate creates a superposition state as a precursor to the rotations.
  
- **Measurement:**  
  After applying the rotations, the qubits are measured. The measurement operations return 1D tensors (of type `tensor<1xi1>`) and a `tensor.extract` is used to obtain scalar values.
  
- **Classical Loop & Optimization:**  
  A classical loop iterates for a fixed number of iterations (5 in this example). In each iteration, the theta values are updated (e.g., incremented by a small constant), and the quantum kernel is called again. This simulates the optimization process in VQE, where the cost function is minimized by adjusting the rotation angles.

---

### 3. Superposition (`superposition.mlir`)
**Description:**  
This example demonstrates the principle of quantum superposition by placing a qubit into an equal superposition state. 

**Detailed Process:**
- **Hadamard Gate Application:**  
  A Hadamard gate is applied to a qubit, transforming it from a definite state |0⟩ into a superposition state |+⟩, where:
|+⟩ = (|0⟩ + |1⟩) / √2

- **Optional CNOT Gate:**  
A CNOT gate is applied between the primary qubit and a second qubit, which can be used to create entanglement if extended further.
- **Measurement:**  
The qubit is measured, with the result returned as a 1D tensor. A `tensor.extract` operation then extracts the scalar measurement value. Over many runs, the probability of obtaining |0⟩ or |1⟩ should be approximately 50% each.

---

### 4. Teleportation (`teleportation.mlir`)
**Description:**  
This example implements the quantum teleportation protocol, which transfers the state of one qubit to another distant qubit using entanglement and classical communication.

**Detailed Process:**
- **Entanglement Generation:**  
Two qubits (`q1` and `q2`) are entangled by applying a Hadamard gate on `q1` followed by a CNOT gate with `q1` as the control and `q2` as the target. This forms a Bell pair.
- **Bell Measurement:**  
The qubit to be teleported (`q0`) is entangled with one half of the Bell pair (`q1`) by applying a CNOT gate followed by a Hadamard gate on `q0`. This constitutes a Bell measurement that projects the combined state onto the Bell basis.
- **Classical Communication & Correction:**  
The measurement outcomes (extracted from 1D tensor outputs using `tensor.extract`) provide two classical bits (`m0` and `m1`). These bits are used to decide which corrective operations to apply to the remaining qubit (`q2`). Specifically:
- If `m1` is true, an X gate is applied.
- If `m0` is true, a Z gate is applied.
- **Verification:**  
A final measurement on `q2` confirms that the state has been successfully teleported.

---

