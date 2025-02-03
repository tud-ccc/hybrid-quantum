set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QIRTransforms)
add_public_tablegen_target(QIRTransformPassesIncGen)

add_mlir_doc(Passes QIRTransformPasses ./ -gen-pass-doc)
