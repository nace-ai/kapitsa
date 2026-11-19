# Concatenates src/kernels/common.h + quant.h + kv_int8.h + src/kernels/hy4/*.metal into one C++ string literal: Hy4-flash's
# kernels, a library of their own compiled on the first Hy4 pipeline (hy4::pipeline, src/model/hy4_kernels.h), so no
# other model's load compiles them. Kernels the base library also has (moe_bucket3) are dispatched from there.
file(READ ${SRC_DIR}/common.h COMMON)
file(READ ${SRC_DIR}/quant.h QUANT)
string(APPEND COMMON "\n${QUANT}")
file(READ ${SRC_DIR}/kv_int8.h KVQ)
string(APPEND COMMON "\n// ---- kv_int8.h ----\n${KVQ}")
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/hy4/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_0-9]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- hy4/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kHy4KernelSource; const char* const kHy4KernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
