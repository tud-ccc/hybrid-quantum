################################################################################
# QILLRIncGen
#
# MLIR QILLR dialect generated includes.
################################################################################

add_custom_target(QILLRIncGen)

# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)
