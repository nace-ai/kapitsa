# Concatenates src/kernels/common.h + src/kernels/dory/*.metal into one C++ string literal: Dory's kernels, a library of
# their own compiled when a Dory model loads (Gpu::compile_extra; its matrix products use the Nemotron-H library).
file(READ ${SRC_DIR}/common.h COMMON)
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/dory/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- dory/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kDoryKernelSource; const char* const kDoryKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
