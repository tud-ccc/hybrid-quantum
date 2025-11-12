add_mlir_upstream_c_api_library(MLIRCAPIQPU
  QPU.cpp

  LINK_LIBS PUBLIC
    QPUIR
    MLIRCAPIIR
)
