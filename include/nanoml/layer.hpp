#pragma once
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <string>

namespace ml {

class Layer {
    public:
        virtual Matrix forward(const Matrix& input) = 0;
        
        virtual Matrix backward(const Matrix& gradient, double learningRate) = 0;
        
        virtual std::string getName() const = 0;
        
        virtual void reset() = 0;
        
        virtual ~Layer() = default;
    };
}