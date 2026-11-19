# Concatenates src/kernels/common.h + quant.h + kv_int8.h + attn.metal + src/kernels/hy4/*.metal into one C++ string literal: Hy4-flash's
# kernels, a library of their own compiled on the first Hy4 pipeline (hy4::pipeline, src/model/hy4_kernels.h), so no
# other model's load compiles them. Kernels the base library also has (moe_bucket3) are dispatched from there.
file(READ ${SRC_DIR}/common.h COMMON)
file(READ ${SRC_DIR}/quant.h QUANT)
string(APPEND COMMON "\n${QUANT}")
file(READ ${SRC_DIR}/kv_int8.h KVQ)
string(APPEND COMMON "\n// ---- kv_int8.h ----\n${KVQ}")
set(ALL "${COMMON}\n")
# attn.metal: hy4_attn_prefill instantiates its attn_prefill11_impl (the base library compiles attn.metal too; this
# copy's kernels are never looked up, pipeline() finds the base library's first)
file(READ ${SRC_DIR}/attn.metal ATTN)
string(REGEX REPLACE "#include \"[a-z_0-9]+\\.h\"" "" ATTN "${ATTN}")
string(APPEND ALL "\n// ---- attn.metal ----\n${ATTN}\n")
file(GLOB METALS ${SRC_DIR}/hy4/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_0-9]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- hy4/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kHy4KernelSource; const char* const kHy4KernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
