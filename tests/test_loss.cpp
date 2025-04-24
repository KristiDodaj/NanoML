#include "nanoml/loss.hpp"
#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

// Helper function for approximate comparison of doubles
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

// Helper function to check if two vectors are approximately equal
bool vectors_approx_equal(const ml::Vector& a, const ml::Vector& b, double epsilon = 1e-9) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], epsilon)) return false;
    }
    return true;
}

int main() {
    // Test MSE (Mean Squared Error)
    {
        // Test case 1: Perfect prediction (y_hat equals y)
        ml::Vector y1(3, 2.0);
        ml::Vector y_hat1(3, 2.0);
        double mse1 = ml::mse(y_hat1, y1);
        assert(approx_equal(mse1, 0.0));
        
        // Test case 2: Simple prediction error
        ml::Vector y2(3, 2.0);
        ml::Vector y_hat2(3, 3.0);  // Each prediction is off by 1.0
        // MSE = (1/2n) * sum((y_hat - y)²) = (1/6) * (3 * (3-2)²) = (1/6) * 3 = 0.5
        double mse2 = ml::mse(y_hat2, y2);
        assert(approx_equal(mse2, 0.5));
        
        // Test case 3: Mixed predictions
        ml::Vector y3(3);
        y3[0] = 1.0; y3[1] = 2.0; y3[2] = 3.0;
        ml::Vector y_hat3(3);
        y_hat3[0] = 2.0; y_hat3[1] = 3.0; y_hat3[2] = 5.0;
        // Differences: 1, 1, 2
        // MSE = (1/2n) * ((1)² + (1)² + (2)²) = (1/6) * (1 + 1 + 4) = (1/6) * 6 = 1.0
        double mse3 = ml::mse(y_hat3, y3);
        assert(approx_equal(mse3, 1.0));
    }

    // Test MSE Gradient
    {
        // Test case 1: Perfect prediction (y_hat equals y)
        ml::Vector y1(3, 2.0);
        ml::Vector y_hat1(3, 2.0);
        ml::Vector grad1 = ml::mse_grad(y_hat1, y1);
        ml::Vector expected_grad1(3, 0.0);  // Gradient is 0 when prediction is perfect
        assert(vectors_approx_equal(grad1, expected_grad1));
        
        // Test case 2: Simple prediction error
        ml::Vector y2(3, 2.0);
        ml::Vector y_hat2(3, 3.0);  // Each prediction is off by 1.0
        ml::Vector grad2 = ml::mse_grad(y_hat2, y2);
        ml::Vector expected_grad2(3, 1.0/3.0);  // (y_hat - y)/n = (3-2)/3 = 1/3
        assert(vectors_approx_equal(grad2, expected_grad2));
        
        // Test case 3: Mixed predictions
        ml::Vector y3(3);
        y3[0] = 1.0; y3[1] = 2.0; y3[2] = 3.0;
        ml::Vector y_hat3(3);
        y_hat3[0] = 2.0; y_hat3[1] = 3.0; y_hat3[2] = 5.0;
        ml::Vector grad3 = ml::mse_grad(y_hat3, y3);
        ml::Vector expected_grad3(3);
        // (y_hat - y)/n
        expected_grad3[0] = (2.0 - 1.0) / 3.0;  // 1/3
        expected_grad3[1] = (3.0 - 2.0) / 3.0;  // 1/3
        expected_grad3[2] = (5.0 - 3.0) / 3.0;  // 2/3
        assert(vectors_approx_equal(grad3, expected_grad3));
    }

    std::cout << "All Loss function tests passed ✅\n";
    return 0;
}