#pragma once
#include <cmath>

namespace ml {
    inline double sigmoid(double z)    { return 1.0 / (1.0 + std::exp(-z)); }
    
    // The correct gradient of sigmoid function with respect to its input
    // When passed a raw input z, we first compute sigmoid(z) and then the gradient
    // When passed an already computed sigmoid output s, we use the direct formula s * (1-s)
    inline double sigmoid_grad(double s) { 
        if (s > 0 && s < 1) {
            return s * (1.0 - s);
        } else {
            double sigmoid_s = sigmoid(s);
            return sigmoid_s * (1.0 - sigmoid_s);
        }
    }
    
    inline double relu(double z) { return z > 0 ? z : 0; }
    
    inline double relu_grad(double z) { return z > 0 ? 1 : 0; }

    inline double tanh_activation(double z) { return std::tanh(z); }
    
    inline double tanh_grad(double z) { 
        double t = std::tanh(z);
        return 1 - t * t; 
    }
}
