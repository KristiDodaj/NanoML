#pragma once
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <string>
#include <fstream>

namespace ml {
    class Model {
        public:
            virtual ~Model() = default;

            virtual Vector forward(const Matrix& X) const = 0;

            virtual void backward(const Matrix& X,
                                const Vector& dLdy,
                                double lr) = 0;
            
            virtual bool save(const std::string& filename) const = 0;
            virtual bool load(const std::string& filename) = 0;
            
            virtual std::string get_model_type() const = 0;
    };
}
