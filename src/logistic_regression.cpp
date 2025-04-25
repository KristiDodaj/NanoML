#include "nanoml/logistic_regression.hpp"
#include "nanoml/activations.hpp"
#include <cassert>

namespace ml {
    LogisticRegression::LogisticRegression(std::size_t n)
    : W(n, 0.0), b(0.0) {}

    Vector LogisticRegression::forward(const Matrix& X) const {
        assert(X.cols() == W.size());
        Vector out(X.rows());
        for (std::size_t i=0;i<X.rows();++i){
            double z = 0.0;
            for (std::size_t k=0;k<X.cols();++k) z += X(i,k)*W[k];
            out[i] = sigmoid(z + b);
        }
        return out;
    }

    void LogisticRegression::backward(const Matrix& X,
                                    const Vector& dLdy,
                                    double lr){
        Vector grad_w(W.size(),0.0);
        double grad_b = 0.0;
        for(std::size_t i=0;i<X.rows();++i){
            grad_b += dLdy[i];
            for(std::size_t k=0;k<X.cols();++k)
                grad_w[k] += X(i,k)*dLdy[i];
        }
        double m = static_cast<double>(X.rows());
        grad_w /= m;  grad_b /= m;

        W -= grad_w * lr;
        b -= grad_b * lr;
    }
}
