add_executable(numpp_tests
    main.cpp
    test_dtype.cpp
    test_manip.cpp
    test_stats.cpp
    test_grids.cpp
    test_indexing.cpp
    test_mathx.cpp
    test_poly.cpp
    test_ndarray.cpp
    test_creation.cpp
    test_backend.cpp
    test_error.cpp
    test_ufunc.cpp
    test_ufunc2.cpp
    test_ufunc3.cpp
    test_linalg.cpp
    test_linalg2.cpp
    test_linalg3.cpp
    test_linalg4.cpp
    test_fft.cpp
    test_fft2.cpp
    test_fft3.cpp
    test_random.cpp
    test_random2.cpp
    test_random3.cpp
    test_random4.cpp
    test_io.cpp
    test_io2.cpp
    test_io3.cpp
    test_sorting.cpp
    test_sorting2.cpp
    test_sorting3.cpp
    test_strings.cpp
    test_datetime.cpp
    test_struct.cpp
    test_gpu.cpp
    test_nanreduce.cpp
)
target_link_libraries(numpp_tests PRIVATE numpp::numpp)
target_include_directories(numpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(numpp_tests PRIVATE
    NUMPP_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
target_compile_options(numpp_tests PRIVATE -Wall -Wextra)

add_test(NAME numpp_tests COMMAND numpp_tests)
