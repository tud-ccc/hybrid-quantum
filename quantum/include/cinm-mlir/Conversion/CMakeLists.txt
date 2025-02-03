set(LLVM_TARGET_DEFINITIONS QuantumPasses.td)
mlir_tablegen(QuantumPasses.h.inc -gen-pass-decls -name QuantumConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(QuantumConversionPassIncGen)
add_mlir_doc(QuantumPasses QuantumConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS QIRPasses.td)
mlir_tablegen(QIRPasses.h.inc -gen-pass-decls -name QIRConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(QIRConversionPassIncGen)
add_mlir_doc(QIRPasses QIRConversionPasses ./ -gen-pass-doc)


