
<br />
<div align="center">
  
  <h3 align="center"> QUMIN: A Compilation Infrastructure for Heterogeneous Quantum Computer </h3>

  <p align="center">
    An MLIR Based Compiler Framework for Quantum Classical Computation (based on Cinnamon)
    <br />
    <a href="https://arxiv.org/abs/2301.07486"><strong>Paper Link»</strong></a>
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

The project aims to develop a comprehensive framework for quantum computing that includes a collection of quantum-specific dialects, enabling targeted backends and progressive lowering to Quantum Intermediate Representation (QIR) and LLVM IR. By implementing optimization passes tailored for the quantum dialect, we seek to enhance the performance and efficiency of quantum algorithms. Additionally, this framework will facilitate the integration of classical dialects, allowing for seamless collaboration between quantum and classical computing paradigms. These features are essential for improving the programmability and usability of quantum architectures, making them more accessible to researchers and developers while promoting interoperability across various quantum platforms.

<!-- 
### Built With

The CINM framework depends on a patched version of LLVM 18.1.6.
Additionally, a number of software packages are required to build it, like CMake.  -->
<!-- 
* [![MLIR][mlir]][Mlir-url]
* [![CMake][CMake]][React-url] -->

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can build the framework locally.

### Prerequisites

QUMIN depends on a patched version of `LLVM 18.1.6`.
Additionally, a number of software packages are required to build it, like `CMake`. 

### Download and Build 

The repository contains a script, `build.sh` that installs all needed dependencies and builds the sources.

* Clone the repo
   ```sh
   git clone https://github.com/tud-ccc/Cinnamon.git
   ```
* Build the sources
   ```sh
   cd Cinnamon
   chmod +x build.sh
   ./build.sh
   ```

<!-- USAGE EXAMPLES -->
## Usage
Run build.sh when running first time. 
Use run.sh when you want to recompile the code after changes. 
The script file test.sh runs a script to check all operation implementation, optimisation pass, lowering to llvm ir etc.  

<!-- LICENSE -->
## License
Distributed under the BSD 3-clause "Clear" License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contributors
- Washim S. Neupane (washim_sharma.neupane@mailbox.tu-dresden.de)

