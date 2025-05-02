cmake_minimum_required(VERSION 3.20)
project(nanoml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nanoml
    src/vector.cpp
    src/matrix.cpp
    src/loss.cpp
    src/metrics.cpp
    src/linear_regression.cpp
    src/logistic_regression.cpp
    src/csv.cpp
    src/neural_network.cpp
    src/tensor_utils.cpp
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

add_executable(test_csv tests/test_csv.cpp)
target_link_libraries(test_csv PRIVATE nanoml)

add_executable(test_model_serialization tests/test_model_serialization.cpp)
target_link_libraries(test_model_serialization PRIVATE nanoml)

add_executable(test_regularization tests/test_regularization.cpp)
target_link_libraries(test_regularization PRIVATE nanoml)

add_executable(test_metrics_validation tests/test_metrics_validation.cpp)
target_link_libraries(test_metrics_validation PRIVATE nanoml)

add_executable(01_vector_matrix_basics examples/01_vector_matrix_basics.cpp)
target_link_libraries(01_vector_matrix_basics PRIVATE nanoml)

add_executable(02_linear_regression examples/02_linear_regression.cpp)
target_link_libraries(02_linear_regression PRIVATE nanoml)

add_executable(03_logistic_regression examples/03_logistic_regression.cpp)
target_link_libraries(03_logistic_regression PRIVATE nanoml)

add_executable(04_neural_network examples/04_neural_network.cpp)
target_link_libraries(04_neural_network PRIVATE nanoml)

add_executable(05_csv_data_handling examples/05_csv_data_handling.cpp)
target_link_libraries(05_csv_data_handling PRIVATE nanoml)

add_executable(06_visualization examples/06_visualization.cpp)
target_link_libraries(06_visualization PRIVATE nanoml)

enable_testing()
add_test(NAME vector COMMAND test_vector)
add_test(NAME matrix COMMAND test_matrix)
add_test(NAME loss COMMAND test_loss)
add_test(NAME linreg COMMAND test_linreg)
add_test(NAME logreg COMMAND test_logreg)
add_test(NAME csv COMMAND test_csv)
add_test(NAME model_serialization COMMAND test_model_serialization)
add_test(NAME regularization COMMAND test_regularization)
add_test(NAME metrics_validation COMMAND test_metrics_validation)
