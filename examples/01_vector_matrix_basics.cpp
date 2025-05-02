#include "nanoml/vector.hpp"
#include "nanoml/matrix.hpp"
#include <iostream>

/**
 * @file 01_vector_matrix_basics.cpp
 * @brief Example demonstrating basic vector and matrix operations in nanoml
 */
int main() {
    std::cout << "NanoML Vector and Matrix Basics Example\n";
    std::cout << "======================================\n\n";
    
    // Vector creation and operations
    std::cout << "Vector Operations:\n";
    std::cout << "-----------------\n";
    
    // Create vectors
    ml::Vector v1(4);
    v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0; v1[3] = 4.0;
    
    ml::Vector v2(4);
    v2[0] = 5.0; v2[1] = 6.0; v2[2] = 7.0; v2[3] = 8.0;
    
    // Display vectors
    std::cout << "v1 = [";
    for (size_t i = 0; i < v1.size(); ++i) {
        std::cout << v1[i];
        if (i < v1.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "v2 = [";
    for (size_t i = 0; i < v2.size(); ++i) {
        std::cout << v2[i];
        if (i < v2.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
    
    // Vector addition
    ml::Vector v_sum = v1 + v2;
    std::cout << "v1 + v2 = [";
    for (size_t i = 0; i < v_sum.size(); ++i) {
        std::cout << v_sum[i];
        if (i < v_sum.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Vector dot product
    double dot_product = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
    }
    std::cout << "v1 · v2 = " << dot_product << "\n\n";
    
    // Matrix operations
    std::cout << "Matrix Operations:\n";
    std::cout << "-----------------\n";
    
    // Create matrix
    ml::Matrix m1(2, 3);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0; m1(0, 2) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0; m1(1, 2) = 6.0;
    
    // Display matrix
    std::cout << "m1 =\n";
    for (size_t i = 0; i < m1.rows(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < m1.cols(); ++j) {
            std::cout << m1(i, j);
            if (j < m1.cols() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
    
    // Matrix transpose
    ml::Matrix m_transpose = m1.transpose();
    std::cout << "m1 transposed =\n";
    for (size_t i = 0; i < m_transpose.rows(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < m_transpose.cols(); ++j) {
            std::cout << m_transpose(i, j);
            if (j < m_transpose.cols() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    return 0;
}