# Concatenates src/kernels/common.h + src/kernels/tensor/*.metal into one C++ string literal: kernels on Metal 4's
# tensor_ops (the M5's GPU matrix units), a library of their own compiled as Metal 4 on a GPU that has them
# (Gpu::compile_extra with metal4).
file(READ ${SRC_DIR}/common.h COMMON)
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/tensor/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- tensor/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kTensorKernelSource; const char* const kTensorKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
