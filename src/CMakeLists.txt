add_library(numpp
    core/dtype.cpp
    core/shape.cpp
    core/ndarray.cpp
    core/creation.cpp
    manip/manip.cpp
    manip/manip_extra.cpp
    stats/stats.cpp
    stats/stats_extra.cpp
    grids/grids.cpp
    construct/construct.cpp
    indexing/indexing.cpp
    indexing/indexing_extra.cpp
    mathx/mathx.cpp
    mathx/typecheck.cpp
    poly/poly.cpp
    poly/poly1d.cpp
    polynomial/polynomial.cpp
    tensor/tensor.cpp
    umath/ufunc.cpp
    linalg/linalg.cpp
    fft/fft.cpp
    random/random.cpp
    random/distributions.cpp
    random/distributions2.cpp
    random/distributions3.cpp
    random/bitgenerators.cpp
    polynomial/orthopoly.cpp
    polynomial/orthopoly_calc.cpp
    strings/char_extra.cpp
    strings/char_extra2.cpp
    io/printopts.cpp
    lib/striding.cpp
    lib/striding2.cpp
    io/npy.cpp
    textio/textio.cpp
    io/npz.cpp
    io/format.cpp
    sorting/sorting.cpp
    sorting/sorting_extra.cpp
    strings/strings.cpp
    strings/char.cpp
    datetime/datetime.cpp
    datetime/busday.cpp
    testing/testing.cpp
    ma/masked.cpp
    struct/struct.cpp
    backend/backend.cpp
)
add_library(numpp::numpp ALIAS numpp)

# Select the BLAS backend translation unit (real vs null) based on the flag.
if(NUMPP_WITH_BLAS)
    target_sources(numpp PRIVATE backend/blas_backend.cpp)
    find_package(BLAS REQUIRED)
    target_link_libraries(numpp PUBLIC BLAS::BLAS)
else()
    target_sources(numpp PRIVATE backend/blas_null.cpp)
endif()

# LAPACK backend (real vs null) — same weak-vtable model as BLAS.
# GPU backend (real CUDA / OpenCL vs CPU-reference 'device' vs null) — weak-vtable model.
if(NUMPP_WITH_CUDA)
    target_sources(numpp PRIVATE backend/cuda_backend.cu)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(numpp PUBLIC CUDA::cudart)
    # PTX-only (virtual) arch: the driver JITs it to the GPU's native arch, so an
    # older nvcc can target a newer GPU (e.g. nvcc 12.0 -> sm_120 / RTX 5060).
    set_target_properties(numpp PROPERTIES CUDA_ARCHITECTURES "70-virtual"
                                           CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
elseif(NUMPP_WITH_OPENCL)
    target_sources(numpp PRIVATE backend/opencl_backend.cpp)
    # Link the OpenCL ICD loader. Prefer a discovered library; fall back to the
    # versioned soname so it works without the -dev symlink/headers.
    find_library(NUMPP_OPENCL_LIB NAMES OpenCL)
    if(NUMPP_OPENCL_LIB)
        target_link_libraries(numpp PUBLIC ${NUMPP_OPENCL_LIB})
    else()
        target_link_options(numpp PUBLIC -l:libOpenCL.so.1)
    endif()
elseif(NUMPP_WITH_REFGPU)
    target_sources(numpp PRIVATE backend/refgpu_backend.cpp)
else()
    target_sources(numpp PRIVATE backend/gpu_null.cpp)
endif()

if(NUMPP_WITH_LAPACK)
    target_sources(numpp PRIVATE backend/lapack_backend.cpp)
    find_package(LAPACK REQUIRED)
    target_link_libraries(numpp PUBLIC LAPACK::LAPACK)
else()
    target_sources(numpp PRIVATE backend/lapack_null.cpp)
endif()

target_compile_definitions(numpp PRIVATE NUMPP_BUILDING)

target_include_directories(numpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)

# Host C++ warnings only — these flags confuse nvcc, so guard by language.
target_compile_options(numpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
if(NUMPP_WARNINGS_AS_ERRORS)
    target_compile_options(numpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()

if(NUMPP_ENABLE_ASAN)
    target_compile_options(numpp PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(numpp PUBLIC -fsanitize=address)
endif()
if(NUMPP_ENABLE_UBSAN)
    target_compile_options(numpp PUBLIC -fsanitize=undefined)
    target_link_options(numpp PUBLIC -fsanitize=undefined)
endif()

set_target_properties(numpp PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# -----------------------------------------------------------------------------
# Install / export (find_package(NumPP) support)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS numpp EXPORT NumPPTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/numpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_BINARY_DIR}/generated/numpp/version.hpp
              ${CMAKE_BINARY_DIR}/generated/numpp/backend/config.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/numpp)
install(EXPORT NumPPTargets NAMESPACE numpp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NumPP)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/NumPPConfig.cmake.in
    ${CMAKE_BINARY_DIR}/NumPPConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NumPP)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/NumPPConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_BINARY_DIR}/NumPPConfig.cmake ${CMAKE_BINARY_DIR}/NumPPConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NumPP)
