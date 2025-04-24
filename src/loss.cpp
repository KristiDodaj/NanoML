#include "nanoml/loss.hpp"
#include <cassert>

namespace ml {

    double mse(const Vector& y_hat, const Vector& y) {
        assert(y_hat.size() == y.size());
        double acc = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i)
            acc += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
        return acc / y.size() * 0.5;          // ½‖e‖²
    }
    
    Vector mse_grad(const Vector& y_hat, const Vector& y) {
        assert(y_hat.size() == y.size());
        Vector g(y.size());
        for (std::size_t i = 0; i < y.size(); ++i)
            g[i] = (y_hat[i] - y[i]) / y.size();   // derivative
        return g;
    }
}

