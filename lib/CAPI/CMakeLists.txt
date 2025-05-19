set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBS)
function(add_mlir_upstream_c_api_library name)
  add_mlir_public_c_api_library(${name} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBS ${name})
endfunction()

add_subdirectory(Dialect)
