#[========================================================================[.rst:
MLIRUtils
---------

Some utility functions for reliably declaring MLIR TableGen targets and
enforcing a common naming scheme.

#]========================================================================]

function(mlir_gen_enums prefix)
    string(TOLOWER ${prefix} filter)
    
    set(LLVM_TARGET_DEFINITIONS ${prefix}Enums.td)

    mlir_tablegen(${prefix}Enums.h.inc -gen-enum-decls)
    mlir_tablegen(${prefix}Enums.cpp.inc -gen-enum-defs)

    add_public_tablegen_target(${prefix}EnumsIncGen)
    add_dependencies(${prefix}IncGen ${prefix}EnumsIncGen)
endfunction()

function(mlir_gen_iface prefix iface kind)
    set(LLVM_TARGET_DEFINITIONS ${iface}.td)

    mlir_tablegen(${iface}.h.inc -gen-${kind}-interface-decls)
    mlir_tablegen(${iface}.cpp.inc -gen-${kind}-interface-defs)

    add_public_tablegen_target(${prefix}${iface}InterfaceIncGen)
    add_dependencies(${prefix}IncGen ${prefix}${iface}InterfaceIncGen)
endfunction()

function(mlir_gen_ir prefix)
    string(TOLOWER ${prefix} filter)

    set(LLVM_TARGET_DEFINITIONS ${prefix}Ops.td)

    mlir_tablegen(${prefix}Base.h.inc -gen-dialect-decls -dialect=${filter})
    mlir_tablegen(${prefix}Base.cpp.inc -gen-dialect-defs -dialect=${filter})
    mlir_tablegen(${prefix}Types.h.inc -gen-typedef-decls -typedefs-dialect=${filter})
    mlir_tablegen(${prefix}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${filter})
    mlir_tablegen(${prefix}Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})
    mlir_tablegen(${prefix}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${filter})
    mlir_tablegen(${prefix}Ops.h.inc -gen-op-decls -dialect=${filter})
    mlir_tablegen(${prefix}Ops.cpp.inc -gen-op-defs -dialect=${filter})

    add_public_tablegen_target(${prefix}IRIncGen)
    add_dependencies(${prefix}IncGen ${prefix}IRIncGen)

    add_mlir_doc(${prefix}Ops ${prefix}Ops Dialects/ -gen-dialect-doc -dialect=${filter})
endfunction()

function(mlir_gen_passes prefix)    
    set(LLVM_TARGET_DEFINITIONS Passes.td)

    mlir_tablegen(Passes.h.inc -gen-pass-decls -name ${prefix})
    mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ${prefix})
    mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ${prefix})

    add_public_tablegen_target(${prefix}PassesIncGen)
    add_dependencies(${prefix}IncGen ${prefix}PassesIncGen)
endfunction()
