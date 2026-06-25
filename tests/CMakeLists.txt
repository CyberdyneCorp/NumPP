add_executable(numpp_tests
    main.cpp
    test_dtype.cpp
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
)
target_link_libraries(numpp_tests PRIVATE numpp::numpp)
target_include_directories(numpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(numpp_tests PRIVATE
    NUMPP_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
target_compile_options(numpp_tests PRIVATE -Wall -Wextra)

add_test(NAME numpp_tests COMMAND numpp_tests)
