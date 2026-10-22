# Concatenates src/kernels/common.h + quant.h + kflash.h + src/kernels/flint/*.metal into one C++ string literal: Flint's kernels,
# a library of their own compiled when a Flint model loads (Gpu::compile_extra; its matrix products use the base
# library's).
file(READ ${SRC_DIR}/common.h COMMON)
file(READ ${SRC_DIR}/quant.h QUANT)
string(APPEND COMMON "\n${QUANT}")
file(READ ${SRC_DIR}/kflash.h KFLASH)
string(APPEND COMMON "\n// ---- kflash.h ----\n${KFLASH}")
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/flint/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- flint/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kFlintKernelSource; const char* const kFlintKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
