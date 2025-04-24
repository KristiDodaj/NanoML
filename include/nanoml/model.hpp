#pragma once
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"

namespace ml {

class Model {
    public:
        virtual ~Model() = default;

        // forward pass: predict ŷ  (size = X.rows)
        virtual Vector forward(const Matrix& X) const = 0;

        // backward pass: update parameters using ∂L/∂ŷ (same length as ŷ)
        virtual void backward(const Matrix& X,
                            const Vector& dLdy,
                            double lr) = 0;
        };
}
