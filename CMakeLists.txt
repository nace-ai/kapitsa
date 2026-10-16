cmake_minimum_required(VERSION 3.24)
project(nace_infer LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
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
set(NI_EMBED ${CMAKE_BINARY_DIR}/generated/kernels_embed.cpp)
add_custom_command(
  OUTPUT ${NI_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUT=${NI_EMBED} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src/kernels
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_kernels.cmake
  DEPENDS ${NI_KERNELS} ${CMAKE_SOURCE_DIR}/cmake/embed_kernels.cmake
  COMMENT "Embedding Metal kernels")

file(GLOB_RECURSE NI_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.mm)
list(FILTER NI_SOURCES EXCLUDE REGEX ".*/src/server/.*")

add_library(naceinfer STATIC ${NI_SOURCES} ${NI_EMBED})
target_include_directories(naceinfer PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(naceinfer PUBLIC "-framework Metal" "-framework Foundation" "-framework CoreFoundation")

file(GLOB NI_TOOLS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/*.cpp ${CMAKE_SOURCE_DIR}/tools/*.mm)
foreach(t ${NI_TOOLS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE naceinfer)
endforeach()

# Kapitsa's server and CLI: `kapitsa serve --model <dir> ...`. build/nace-infer-server stays as a
# symlink to it for scripts written before the rename.
add_executable(kapitsa ${CMAKE_SOURCE_DIR}/src/server/main.cpp)
target_include_directories(kapitsa PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(kapitsa PRIVATE naceinfer "-framework CoreFoundation")
add_custom_command(TARGET kapitsa POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/nace-infer-server
  COMMAND ${CMAKE_COMMAND} -E create_symlink kapitsa ${CMAKE_BINARY_DIR}/nace-infer-server)
