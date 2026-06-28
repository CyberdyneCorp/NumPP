# Consumes NumPP from an INSTALLED prefix via find_package (regression guard for
# #107: installed umbrella header must compile). Point CMAKE_PREFIX_PATH at the
# install prefix when configuring this project.
cmake_minimum_required(VERSION 3.25)
project(numpp_find_package_consumer LANGUAGES CXX)

find_package(NumPP REQUIRED)

add_executable(consumer ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
target_compile_features(consumer PRIVATE cxx_std_20)
target_link_libraries(consumer PRIVATE numpp::numpp)
