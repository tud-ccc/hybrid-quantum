################################################################################
# QIRIncGen
#
# MLIR QIR dialect generated includes.
################################################################################

add_custom_target(QIRIncGen)

# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
add_subdirectory(Transforms)
