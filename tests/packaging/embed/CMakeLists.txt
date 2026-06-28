# Embeds NumPP via add_subdirectory (regression guard for #108: install/export
# rules must not assume NumPP is the top-level project). Pass the NumPP source
# tree as -DNUMPP_SOURCE_DIR=<repo-root> when configuring.
cmake_minimum_required(VERSION 3.25)
project(numpp_embed_consumer LANGUAGES CXX)

if(NOT DEFINED NUMPP_SOURCE_DIR)
    message(FATAL_ERROR "Pass -DNUMPP_SOURCE_DIR=<path-to-numpp-repo>")
endif()

# Embedded builds don't need the test suite.
set(NUMPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${NUMPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/numpp_build)

add_executable(embed ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
target_compile_features(embed PRIVATE cxx_std_20)
target_link_libraries(embed PRIVATE numpp::numpp)
