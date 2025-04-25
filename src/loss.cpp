#include "nanoml/loss.hpp"
#include <cassert>
#include <cmath>
#include <algorithm> // Required for std::clamp

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

    double clip(double x){ return std::clamp(x, 1e-12, 1.0-1e-12); }

    double bce(const ml::Vector& y_hat, const ml::Vector& y){
        assert(y_hat.size() == y.size());
        double acc=0.0;
        for(std::size_t i=0;i<y.size();++i){
            double p = clip(y_hat[i]);
            acc += - ( y[i]*std::log(p) + (1-y[i])*std::log(1-p) );
        }
        return acc / y.size();
    }

    ml::Vector bce_grad(const ml::Vector& y_hat, const ml::Vector& y){
        assert(y_hat.size() == y.size());
        ml::Vector g(y.size());
        for(std::size_t i=0;i<y.size();++i){
            double p = clip(y_hat[i]);
            g[i] = (p - y[i]) / y.size();  // derivative wrt p
        }
        return g;
    }
}

