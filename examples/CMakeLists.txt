# Build every example program. Each links the library and self-verifies against
# NumPy via parity.hpp (in this directory).
file(GLOB EXAMPLE_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(src ${EXAMPLE_SRCS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(numpp_${name} ${src})
    target_link_libraries(numpp_${name} PRIVATE numpp::numpp)
    target_include_directories(numpp_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
