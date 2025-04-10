# Quantum Algorithms in MLIR Quantum Dialect

This folder demonstrates several foundational quantum algorithms implemented using our custom MLIR quantum dialect. The following algorithms are included:

- **Grover's Algorithm**
- **Deutsch Algorithm**
- **Bell Pair (EPR Pair) Generation**
- **Quantum Teleportation**

Each algorithm leverages the core operations available in the dialect (such as Hadamard, CNOT, Pauli gates, allocation, splitting/merging of qubits, and measurement) to illustrate key quantum computing principles. Some simplifications are made since this is just for demonstration purpose. 

---

## Grover's Algorithm

**Purpose:**  
Grover's algorithm provides a quadratic speedup for unstructured search problems by finding a marked element in an unsorted database.

**How It Works:**  
- **Initialization:** Two qubits are allocated and prepared in a uniform superposition using Hadamard gates.
- **Oracle:** A phase inversion is applied to the marked state (for example, `|11⟩`), effectively "marking" the target element.
- **Diffusion Operator:** Inversion about the mean is performed to amplify the probability amplitude of the marked state.
- **Measurement:** The final state is measured, revealing the marked element with high probability.

This algorithm illustrates the power of quantum parallelism and interference, making it a cornerstone for quantum search applications.

---

## Deutsch Algorithm

**Purpose:**  
The Deutsch algorithm is one of the simplest quantum algorithms, designed to determine whether a given binary function is constant or balanced with a single function evaluation.

**How It Works:**  
- **Initialization:** Two qubits are used; one is prepared as the input (in the state `|0⟩`), and the other is an ancilla (initialized to `|1⟩` via an X gate).
- **Superposition:** Both qubits are put into a superposition using Hadamard gates.
- **Oracle Application:** The oracle (representing the function) is implemented using a CNOT or equivalent gate sequence. It flips the ancilla based on the input value.
- **Interference:** A second Hadamard gate is applied to the input qubit to cause interference, effectively collapsing the superposition based on the oracle’s behavior.
- **Measurement:** The resulting state of the input qubit is measured. An outcome of `0` indicates a constant function, while `1` signals a balanced function.

This algorithm demonstrates quantum parallelism and interference, enabling function characterization with fewer queries than classical approaches.

---

## Bell Pair (EPR Pair) Generation

**Purpose:**  
Bell pairs, or EPR pairs, are maximally entangled qubit pairs that form the basis of many quantum communication protocols, such as quantum key distribution and teleportation.

**How It Works:**  
- **Allocation and Splitting:** Two qubits are allocated together and then split into individual qubits.
- **Entanglement Creation:** A Hadamard gate is applied to the first qubit to generate superposition, and a subsequent CNOT gate (with the first qubit as control and the second as target) entangles the two qubits.
- **Measurement (Optional):** Measuring the entangled qubits shows correlated outcomes (either both `0` or both `1`), confirming their entangled nature.

Creating Bell pairs is fundamental for exploring non-local correlations and serves as a building block for more complex protocols like teleportation.

---

## Quantum Teleportation

**Purpose:**  
Quantum teleportation transfers the state of a qubit from one location (Alice) to another (Bob) using entanglement and classical communication—without physically moving the qubit itself.

**How It Works:**  
- **Preparation:** Three qubits are used:
  - **Qubit A:** The qubit with the state to be teleported.
  - **Qubits B and C:** Form an entangled Bell pair shared between Alice and Bob.
- **Entanglement:** Qubits B and C are entangled using a Hadamard and a CNOT gate.
- **Bell Measurement (Alice’s Side):**  
  - Alice performs a Bell measurement on Qubit A and her half of the Bell pair (Qubit B) by applying a CNOT (with A as control) followed by a Hadamard on A.
  - Both qubits are measured, producing two classical bits.
- **Classical Communication and Correction (Bob’s Side):**  
  - The measurement outcomes are sent to Bob.
  - Based on these outcomes, Bob applies conditional corrections (using X and Z gates) to his qubit (Qubit C) to reconstruct the original state from Qubit A.
- **Verification:**  
  - Bob’s corrected qubit is measured to confirm that the state has been successfully teleported.

This protocol highlights the interplay between quantum entanglement and classical communication, demonstrating non-local transfer of quantum information.

---

