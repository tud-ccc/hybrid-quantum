set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QuantumTransforms)
add_public_tablegen_target(QuantumTransformPassesIncGen)

add_mlir_doc(Passes QuantumTransformPasses ./ -gen-pass-doc)
