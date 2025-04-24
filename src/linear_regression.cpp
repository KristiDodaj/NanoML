#include "nanoml/linear_regression.hpp"
#include <cassert>

namespace  ml {

    LinearRegression::LinearRegression(std::size_t n_features)
        : W(n_features, 0.0), b(0.0) {}

    Vector LinearRegression::forward(const Matrix& X) const {
        assert(X.cols() == W.size());
        Vector y_hat(X.rows(), 0.0);
        for (std::size_t i = 0; i < X.rows(); ++i) {
            double dot = 0.0;
            for (std::size_t k = 0; k < X.cols(); ++k)
                dot += X(i,k) * W[k];
            y_hat[i] = dot + b;
        }
        return y_hat;
    }

    void LinearRegression::backward(const Matrix& X,
                                     const Vector& dLdy,
                                     double lr) {
        assert(X.rows() == dLdy.size());
        assert(X.cols() == W.size());

        // Update weights
        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t k = 0; k < X.cols(); ++k)
                W[k] -= lr * dLdy[i] * X(i, k);
        }

        // Update bias
        for (std::size_t i = 0; i < X.rows(); ++i)
            b -= lr * dLdy[i];
    }

}