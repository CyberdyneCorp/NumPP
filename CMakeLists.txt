cmake_minimum_required(VERSION 3.25)

project(NumPP
    VERSION 0.1.0
    DESCRIPTION "Modern C++20 port of NumPy with tiered acceleration"
    HOMEPAGE_URL "https://github.com/leonardoaraujosantos/NumPP"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(NUMPP_BUILD_SHARED       "Build shared library"              ON)
option(NUMPP_BUILD_TESTS        "Build test suite"                  ON)
option(NUMPP_BUILD_EXAMPLES     "Build examples"                    OFF)
option(NUMPP_BUILD_BENCHMARKS   "Build benchmarks"                  OFF)
option(NUMPP_BUILD_DOCS         "Build Doxygen API documentation"   OFF)
option(NUMPP_ENABLE_ASAN        "Enable AddressSanitizer"           OFF)
option(NUMPP_ENABLE_UBSAN       "Enable UndefinedBehaviorSanitizer" OFF)
option(NUMPP_WARNINGS_AS_ERRORS "Treat warnings as errors"          ON)

# Optional acceleration backends — all OFF by default. A build with every flag
# OFF is the portable mobile (iOS/Android) baseline and must be fully functional.
option(NUMPP_WITH_BLAS          "Enable optional BLAS backend"      OFF)
option(NUMPP_WITH_LAPACK        "Enable optional LAPACK backend"    OFF)
option(NUMPP_WITH_METAL         "Enable optional Metal GPU backend" OFF)
option(NUMPP_WITH_VULKAN        "Enable optional Vulkan GPU backend" OFF)
option(NUMPP_WITH_CUDA          "Enable optional CUDA GPU backend"  OFF)
option(NUMPP_WITH_OPENCL        "Enable optional OpenCL GPU backend" OFF)
option(NUMPP_WITH_REFGPU        "Enable CPU-reference GPU device (testing)" OFF)

# -----------------------------------------------------------------------------
# Project-wide settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NUMPP_BUILD_SHARED)
    set(BUILD_SHARED_LIBS ON)
endif()

# -----------------------------------------------------------------------------
# Toolchain guard (build-packaging: fail early with a clear message)
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "NumPP requires clang >= 14 for C++20 (found ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "NumPP requires GCC >= 11 for C++20 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# -----------------------------------------------------------------------------
# Generated headers
# -----------------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/numpp/backend/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/numpp/backend/config.hpp" @ONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/numpp/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/numpp/version.hpp" @ONLY)

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------
add_subdirectory(src)

if(NUMPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(NUMPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
