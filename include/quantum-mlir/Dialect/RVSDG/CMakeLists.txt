################################################################################
# RVSDGIncGen
#
# MLIR RVSDG dialect generated includes.
################################################################################

add_custom_target(RVSDGIncGen)

#mlir_gen_enums(RVSDG)

# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
#add_subdirectory(Transforms)
