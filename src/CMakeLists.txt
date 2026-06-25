add_library(numpp
    core/dtype.cpp
    core/shape.cpp
    core/ndarray.cpp
    core/creation.cpp
    manip/manip.cpp
    stats/stats.cpp
    grids/grids.cpp
    indexing/indexing.cpp
    mathx/mathx.cpp
    poly/poly.cpp
    umath/ufunc.cpp
    linalg/linalg.cpp
    fft/fft.cpp
    random/random.cpp
    io/npy.cpp
    io/npz.cpp
    io/format.cpp
    sorting/sorting.cpp
    strings/strings.cpp
    datetime/datetime.cpp
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
# GPU backend (CPU-reference 'device' vs null) — weak-vtable model.
if(NUMPP_WITH_REFGPU)
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

target_compile_options(numpp PRIVATE -Wall -Wextra -Wpedantic)
if(NUMPP_WARNINGS_AS_ERRORS)
    target_compile_options(numpp PRIVATE -Werror)
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
