cmake_minimum_required(VERSION 3.20)
project(nanoml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nanoml
    src/vector.cpp
    src/matrix.cpp
    src/loss.cpp
    src/linear_regression.cpp
    src/logistic_regression.cpp
)

target_include_directories(nanoml PUBLIC include)

add_executable(test_vector tests/test_vector.cpp)
target_link_libraries(test_vector PRIVATE nanoml)

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE nanoml)

add_executable(test_loss tests/test_loss.cpp)
target_link_libraries(test_loss PRIVATE nanoml)

add_executable(test_linreg tests/test_linreg.cpp)
target_link_libraries(test_linreg PRIVATE nanoml)

add_executable(test_logreg tests/test_logreg.cpp)
target_link_libraries(test_logreg PRIVATE nanoml)

enable_testing()
add_test(NAME vector COMMAND test_vector)
add_test(NAME matrix COMMAND test_matrix)
add_test(NAME loss COMMAND test_loss)
add_test(NAME linreg COMMAND test_linreg)
add_test(NAME logreg COMMAND test_logreg)
