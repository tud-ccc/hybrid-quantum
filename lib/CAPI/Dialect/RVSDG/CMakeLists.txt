add_mlir_upstream_c_api_library(MLIRCAPIRVSDG
  RVSDG.cpp

  LINK_LIBS PUBLIC
    RVSDGIR
    MLIRCAPIIR
)
