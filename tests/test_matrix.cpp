#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>
#include <cmath> // For std::fabs

// Helper function for approximate comparison of doubles
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    // Test Constructor and operator()
    ml::Matrix A(2, 3, 1.0); // 2x3 matrix filled with 1.0
    assert(A.rows() == 2);
    assert(A.cols() == 3);
    assert(A.size() == 6);
    assert(approx_equal(A(0, 0), 1.0));
    assert(approx_equal(A(1, 2), 1.0));

    // Test non-const operator()
    A(0, 1) = 5.0;
    assert(approx_equal(A(0, 1), 5.0));

    ml::Matrix B(2, 3, 2.0); // 2x3 matrix filled with 2.0
    assert(B.rows() == 2);
    assert(B.cols() == 3);

    // Test operator+
    ml::Matrix C = A + B; // A = {{1, 5, 1}, {1, 1, 1}}, B = {{2, 2, 2}, {2, 2, 2}}
    assert(C.rows() == 2);
    assert(C.cols() == 3);
    assert(approx_equal(C(0, 0), 3.0)); // 1 + 2
    assert(approx_equal(C(0, 1), 7.0)); // 5 + 2
    assert(approx_equal(C(0, 2), 3.0)); // 1 + 2
    assert(approx_equal(C(1, 0), 3.0)); // 1 + 2
    assert(approx_equal(C(1, 1), 3.0)); // 1 + 2
    assert(approx_equal(C(1, 2), 3.0)); // 1 + 2

    // Test operator+=
    ml::Matrix D(2, 3, 0.5);
    D += A; // D = {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}} + {{1, 5, 1}, {1, 1, 1}}
    assert(approx_equal(D(0, 0), 1.5));
    assert(approx_equal(D(0, 1), 5.5));
    assert(approx_equal(D(0, 2), 1.5));
    assert(approx_equal(D(1, 0), 1.5));
    assert(approx_equal(D(1, 1), 1.5));
    assert(approx_equal(D(1, 2), 1.5));

    // Test operator-
    ml::Matrix E = C - A; // C = {{3, 7, 3}, {3, 3, 3}}, A = {{1, 5, 1}, {1, 1, 1}}
    assert(E.rows() == 2);
    assert(E.cols() == 3);
    assert(approx_equal(E(0, 0), 2.0)); // 3 - 1
    assert(approx_equal(E(0, 1), 2.0)); // 7 - 5
    assert(approx_equal(E(0, 2), 2.0)); // 3 - 1
    assert(approx_equal(E(1, 0), 2.0)); // 3 - 1
    assert(approx_equal(E(1, 1), 2.0)); // 3 - 1
    assert(approx_equal(E(1, 2), 2.0)); // 3 - 1
    // Check if E is now equal to B
    assert(approx_equal(E(0, 0), B(0, 0)));
    assert(approx_equal(E(1, 2), B(1, 2)));

    // Test operator-=
    ml::Matrix F_mat = C; // F_mat = {{3, 7, 3}, {3, 3, 3}}
    F_mat -= A; // F_mat = {{3, 7, 3}, {3, 3, 3}} - {{1, 5, 1}, {1, 1, 1}}
    assert(approx_equal(F_mat(0, 0), 2.0));
    assert(approx_equal(F_mat(1, 2), 2.0));
    // Check if F_mat is now equal to B
    assert(approx_equal(F_mat(0, 0), B(0, 0)));
    assert(approx_equal(F_mat(1, 2), B(1, 2)));

    // Test operator* (scalar)
    ml::Matrix G_mat = A * 3.0; // A = {{1, 5, 1}, {1, 1, 1}}
    assert(G_mat.rows() == 2);
    assert(G_mat.cols() == 3);
    assert(approx_equal(G_mat(0, 0), 3.0));  // 1 * 3
    assert(approx_equal(G_mat(0, 1), 15.0)); // 5 * 3
    assert(approx_equal(G_mat(0, 2), 3.0));  // 1 * 3
    assert(approx_equal(G_mat(1, 0), 3.0));  // 1 * 3
    assert(approx_equal(G_mat(1, 1), 3.0));  // 1 * 3
    assert(approx_equal(G_mat(1, 2), 3.0));  // 1 * 3

    // Test operator*= (scalar)
    ml::Matrix H_mat = A; // H_mat = {{1, 5, 1}, {1, 1, 1}}
    H_mat *= 3.0;
    assert(approx_equal(H_mat(0, 0), 3.0));
    assert(approx_equal(H_mat(0, 1), 15.0));
    assert(approx_equal(H_mat(1, 2), 3.0));
    // Check if H_mat is now equal to G_mat
    assert(approx_equal(H_mat(0, 0), G_mat(0, 0)));
    assert(approx_equal(H_mat(1, 2), G_mat(1, 2)));

    // Test operator/ (scalar)
    ml::Matrix I_mat = B / 2.0; // B = {{2, 2, 2}, {2, 2, 2}}
    assert(I_mat.rows() == 2);
    assert(I_mat.cols() == 3);
    assert(approx_equal(I_mat(0, 0), 1.0)); // 2 / 2
    assert(approx_equal(I_mat(1, 2), 1.0)); // 2 / 2

    // Test operator/= (scalar)
    ml::Matrix J_mat = B; // J_mat = {{2, 2, 2}, {2, 2, 2}}
    J_mat /= 2.0;
    assert(approx_equal(J_mat(0, 0), 1.0));
    assert(approx_equal(J_mat(1, 2), 1.0));
    // Check if J_mat is now equal to I_mat
    assert(approx_equal(J_mat(0, 0), I_mat(0, 0)));
    assert(approx_equal(J_mat(1, 2), I_mat(1, 2)));

    // Test Matrix-Vector multiplication
    // A = {{1, 5, 1}, {1, 1, 1}}
    ml::Vector v(3, 0.0);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0; // v = {1, 2, 3}
    ml::Vector Av = A * v;
    assert(Av.size() == 2);
    // Row 0: (1*1) + (5*2) + (1*3) = 1 + 10 + 3 = 14
    // Row 1: (1*1) + (1*2) + (1*3) = 1 + 2 + 3 = 6
    assert(approx_equal(Av[0], 14.0));
    assert(approx_equal(Av[1], 6.0));

    // Test Matrix-Matrix multiplication
    ml::Matrix K(3, 2, 0.0); // 3x2 matrix
    K(0, 0) = 1; K(0, 1) = 2;
    K(1, 0) = 3; K(1, 1) = 4;
    K(2, 0) = 5; K(2, 1) = 6;
    // K = {{1, 2}, {3, 4}, {5, 6}}
    // A = {{1, 5, 1}, {1, 1, 1}} (2x3)
    ml::Matrix AK = A * K; // Should be 2x2
    assert(AK.rows() == 2);
    assert(AK.cols() == 2);
    // AK(0,0) = (1*1)+(5*3)+(1*5) = 1 + 15 + 5 = 21
    // AK(0,1) = (1*2)+(5*4)+(1*6) = 2 + 20 + 6 = 28
    // AK(1,0) = (1*1)+(1*3)+(1*5) = 1 + 3 + 5 = 9
    // AK(1,1) = (1*2)+(1*4)+(1*6) = 2 + 4 + 6 = 12
    assert(approx_equal(AK(0, 0), 21.0));
    assert(approx_equal(AK(0, 1), 28.0));
    assert(approx_equal(AK(1, 0), 9.0));
    assert(approx_equal(AK(1, 1), 12.0));

    // Test transpose
    // A = {{1, 5, 1}, {1, 1, 1}} (2x3)
    ml::Matrix At = A.transpose(); // Should be 3x2
    assert(At.rows() == 3);
    assert(At.cols() == 2);
    // At = {{1, 1}, {5, 1}, {1, 1}}
    assert(approx_equal(At(0, 0), A(0, 0)));
    assert(approx_equal(At(0, 1), A(1, 0)));
    assert(approx_equal(At(1, 0), A(0, 1)));
    assert(approx_equal(At(1, 1), A(1, 1)));
    assert(approx_equal(At(2, 0), A(0, 2)));
    assert(approx_equal(At(2, 1), A(1, 2)));


    std::cout << "All Matrix tests passed ✅\n";
    return 0;
}
