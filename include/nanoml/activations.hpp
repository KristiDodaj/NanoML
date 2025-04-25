#pragma once
#include <cmath>

namespace ml {
    inline double sigmoid(double z)    { return 1.0 / (1.0 + std::exp(-z)); }
    inline double sigmoid_grad(double s) { return s * (1.0 - s); }
} 
