cmake_minimum_required(VERSION 3.24)
# before project(): CMake reads the deployment target when it sets up the compilers
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
project(nace_infer LANGUAGES C CXX OBJCXX)

set(KAPITSA_VERSION "0.2.9")
if(NOT DEFINED KAPITSA_GIT)  # packaging/release/build.sh passes the commit it exported
  execute_process(COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short HEAD OUTPUT_VARIABLE KAPITSA_GIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3")
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# Metal kernel sources are embedded as strings and compiled at runtime (no offline
# `metal` compiler needed; works with Command Line Tools only).
file(GLOB NI_KERNELS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/*.metal ${CMAKE_SOURCE_DIR}/src/kernels/*.h)
# (src/kernels/kflash.h is in NI_KERNELS: both embedded libraries read it)
set(NI_EMBED ${CMAKE_BINARY_DIR}/generated/kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels.cmake
  DEPENDS ${NI_KERNELS} ${CMAKE_SOURCE_DIR}/cmake/embed_kernels.cmake
  COMMENT "Embedding Metal kernels")

# Nemotron-H kernels: a second embedded source, compiled only when such a model loads.
file(GLOB NI_NH_KERNELS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/nemotron/*.metal ${CMAKE_SOURCE_DIR}/src/kernels/nemotron/*.h)
set(NI_NH_EMBED ${CMAKE_BINARY_DIR}/generated/nh_kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_NH_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_NH_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_nh.cmake
  DEPENDS ${NI_NH_KERNELS} ${CMAKE_SOURCE_DIR}/src/kernels/common.h ${CMAKE_SOURCE_DIR}/src/kernels/iq.h ${CMAKE_SOURCE_DIR}/src/kernels/kflash.h
          ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_nh.cmake
  COMMENT "Embedding the Nemotron-H Metal kernels")

# Dory kernels: a third embedded source, compiled only when a Dory model loads.
file(GLOB NI_DORY_KERNELS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/dory/*.metal)
set(NI_DORY_EMBED ${CMAKE_BINARY_DIR}/generated/dory_kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_DORY_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_DORY_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_dory.cmake
  DEPENDS ${NI_DORY_KERNELS} ${CMAKE_SOURCE_DIR}/src/kernels/common.h ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_dory.cmake
  COMMENT "Embedding the Dory Metal kernels")

# Flint kernels: compiled only when a Flint model loads.
file(GLOB NI_FLINT_KERNELS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/flint/*.metal)
set(NI_FLINT_EMBED ${CMAKE_BINARY_DIR}/generated/flint_kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_FLINT_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_FLINT_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_flint.cmake
  DEPENDS ${NI_FLINT_KERNELS} ${CMAKE_SOURCE_DIR}/src/kernels/common.h ${CMAKE_SOURCE_DIR}/src/kernels/quant.h
          ${CMAKE_SOURCE_DIR}/src/kernels/kflash.h
          ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_flint.cmake
  COMMENT "Embedding the Flint Metal kernels")

# Metal 4 tensor_ops kernels (src/kernels/tensor/): a library of their own, compiled as Metal 4 on a GPU with matrix units
file(GLOB NI_TENSOR_KERNELS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/tensor/*.metal)
set(NI_TENSOR_EMBED ${CMAKE_BINARY_DIR}/generated/tensor_kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_TENSOR_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_TENSOR_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_tensor.cmake
  DEPENDS ${NI_TENSOR_KERNELS} ${CMAKE_SOURCE_DIR}/src/kernels/common.h ${CMAKE_SOURCE_DIR}/src/kernels/nemotron/nh_route.h
          ${CMAKE_SOURCE_DIR}/src/kernels/iq.h
          ${CMAKE_SOURCE_DIR}/cmake/embed_kernels_tensor.cmake
  COMMENT "Embedding the Metal 4 tensor kernels")

# kapitsa bench (src/bench/): the harness's frozen prompt corpus and Apple's names for the Mac models, embedded
set(NI_BENCH_CORPUS ${CMAKE_BINARY_DIR}/generated/bench_corpus.cpp)
add_custom_command(
  OUTPUT ${NI_BENCH_CORPUS}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_BENCH_CORPUS} -DSRC=${CMAKE_SOURCE_DIR}/bench/harness/corpus/bench_corpus.txt
          -DNAME=kBenchCorpus -P ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  DEPENDS ${CMAKE_SOURCE_DIR}/bench/harness/corpus/bench_corpus.txt ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  COMMENT "Embedding the benchmark's prompt corpus")
set(NI_MAC_MODELS ${CMAKE_BINARY_DIR}/generated/mac_models.cpp)
add_custom_command(
  OUTPUT ${NI_MAC_MODELS}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_MAC_MODELS} -DSRC=${CMAKE_SOURCE_DIR}/bench/community/mac-models.json
          -DNAME=kMacModels -P ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  DEPENDS ${CMAKE_SOURCE_DIR}/bench/community/mac-models.json ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  COMMENT "Embedding the Mac models")

file(GLOB_RECURSE NI_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.mm)
list(FILTER NI_SOURCES EXCLUDE REGEX ".*/src/server/.*")

add_library(naceinfer STATIC ${NI_SOURCES} ${NI_EMBED} ${NI_NH_EMBED} ${NI_DORY_EMBED} ${NI_FLINT_EMBED} ${NI_TENSOR_EMBED} ${NI_BENCH_CORPUS} ${NI_MAC_MODELS})
target_include_directories(naceinfer PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(naceinfer PUBLIC "-framework Metal" "-framework Foundation" "-framework CoreFoundation" "-framework IOSurface"
  "-framework IOKit" "-framework ImageIO" "-framework CoreGraphics" "-framework Security")
target_compile_definitions(naceinfer PRIVATE KAPITSA_VERSION="${KAPITSA_VERSION}")

file(GLOB NI_TOOLS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/*.cpp ${CMAKE_SOURCE_DIR}/tools/*.mm)
# a library the Python converters build and load themselves (ctypes), not a command
list(FILTER NI_TOOLS EXCLUDE REGEX ".*/tools/mlx_quant\\.cpp$")
foreach(t ${NI_TOOLS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE naceinfer)
  # (KAPITSA_GIT: the commit a dump or measurement says it was built from, as `kapitsa --version` does)
  target_compile_definitions(${name} PRIVATE KAPITSA_VERSION="${KAPITSA_VERSION}" KAPITSA_GIT="${KAPITSA_GIT}")
endforeach()

# Kapitsa's server and CLI: `kapitsa serve --model <name|dir> ...`, with the chat page (src/server/chat.html)
# embedded, the coding-agent launchers (src/server/launch.cpp) and `kapitsa update` (src/server/update_cmd.cpp). The API
# translations (src/api/) and the updater (src/update/) are in the library.
# build/nace-infer-server stays as a symlink to it for scripts written before the rename.
set(KAPITSA_CHAT ${CMAKE_BINARY_DIR}/generated/chat_page.cpp)
add_custom_command(
  OUTPUT ${KAPITSA_CHAT}
  COMMAND ${CMAKE_COMMAND} -DOUT=${KAPITSA_CHAT} -DSRC=${CMAKE_SOURCE_DIR}/src/server/chat.html -DNAME=kChatPage
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  DEPENDS ${CMAKE_SOURCE_DIR}/src/server/chat.html ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  COMMENT "Embedding the chat page")
add_executable(kapitsa ${CMAKE_SOURCE_DIR}/src/server/main.cpp ${CMAKE_SOURCE_DIR}/src/server/launch.cpp
  ${CMAKE_SOURCE_DIR}/src/server/ui.cpp ${CMAKE_SOURCE_DIR}/src/server/beam.cpp ${CMAKE_SOURCE_DIR}/src/server/picker.cpp
  ${CMAKE_SOURCE_DIR}/src/server/bench_cmd.cpp ${CMAKE_SOURCE_DIR}/src/server/update_cmd.cpp ${KAPITSA_CHAT})
target_include_directories(kapitsa PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(kapitsa PRIVATE naceinfer "-framework CoreFoundation")
target_compile_definitions(kapitsa PRIVATE KAPITSA_VERSION="${KAPITSA_VERSION}" KAPITSA_GIT="${KAPITSA_GIT}")
add_custom_command(TARGET kapitsa POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/nace-infer-server
  COMMAND ${CMAKE_COMMAND} -E create_symlink kapitsa ${CMAKE_BINARY_DIR}/nace-infer-server)
