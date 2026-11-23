# Concatenates src/kernels/common.h + src/kernels/inkling2/*.metal into one C++ string literal:
# Inkling-2 kernels compiled once lazily via Gpu::compile_extra("Ink2", kInk2KernelSource).
file(READ ${SRC_DIR}/common.h COMMON)
file(READ ${SRC_DIR}/inkling2/gguf_blocks.h GGUF_BLOCKS)
set(ALL "${COMMON}\n${GGUF_BLOCKS}\n")
file(GLOB METALS ${SRC_DIR}/inkling2/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- inkling2/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kInk2KernelSource; const char* const kInk2KernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
