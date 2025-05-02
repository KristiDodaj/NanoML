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
            
            void update_weight(std::size_t idx, double delta) { W[idx] += delta; }
            void update_bias(double delta) { b += delta; }

            bool save(const std::string& filename) const override;
            bool load(const std::string& filename) override;
            std::string get_model_type() const override { return "LinearRegression"; }

        private:
            Vector W;
            double b;
    };
}
