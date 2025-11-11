add_mlir_upstream_c_api_library(MLIRCAPIQUANTUM
  Quantum.cpp

  LINK_LIBS PUBLIC
    QuantumIR
    MLIRCAPIIR
)
