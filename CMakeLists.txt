cmake_minimum_required(VERSION 3.20)
project(nanoml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nanoml
    src/vector.cpp
)

target_include_directories(nanoml PUBLIC include)

add_executable(test_vector tests/test_vector.cpp)
target_link_libraries(test_vector PRIVATE nanoml)

enable_testing()
add_test(NAME vector COMMAND test_vector)
