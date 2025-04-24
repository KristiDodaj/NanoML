#pragma once
#include "nanoml/model.hpp"

namespace ml {

    class LinearRegression : public Model {
        public:
            explicit LinearRegression(std::size_t n_features);

            Vector forward(const Matrix& X) const override;
            void   backward(const Matrix& X,
                            const Vector& dLdy,
                            double lr) override;

        private:
            Vector W;   // weight vector (n_features)
            double b;   // bias
    };

}
