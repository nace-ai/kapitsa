# Concatenates src/kernels/common.h, quant.h, iq.h, kflash.h, kv_int8.h + every *.metal into one C++ string literal.
file(READ ${SRC_DIR}/common.h COMMON)
file(READ ${SRC_DIR}/quant.h QUANT)
file(READ ${SRC_DIR}/iq.h IQ)
string(APPEND COMMON "\n${QUANT}\n${IQ}")
file(READ ${SRC_DIR}/kflash.h KFLASH)
string(APPEND COMMON "\n// ---- kflash.h ----\n${KFLASH}")
file(READ ${SRC_DIR}/kv_int8.h KVQ)
string(APPEND COMMON "\n// ---- kv_int8.h ----\n${KVQ}")
file(GLOB METALS ${SRC_DIR}/*.metal)
list(SORT METALS)
set(ALL "${COMMON}\n")
# Format lookup checks the base library first. These units contain base entry
# points and supply no declarations needed by the format instantiations.
set(BASE_ONLY
  clover.metal exl3moe.metal qwen4.metal verify8_fuse.metal
  verify8_packed_coeff.metal vit.metal zq2moe.metal zqwen4v.metal)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"(common|kv_int8)\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  list(FIND BASE_ONLY "${n}" base_only_index)
  set(SECTION "\n// ---- ${n} ----\n${BODY}\n")
  if(NOT base_only_index EQUAL -1)
    set(SECTION "\n#ifndef NI_WF_INST\n${SECTION}#endif\n")
  endif()
  string(APPEND ALL "${SECTION}")
endforeach()
# Split into chunks: MSVC-style literal limits don't apply to clang, but keep delimiters unique.
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kKernelSource; const char* const kKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
