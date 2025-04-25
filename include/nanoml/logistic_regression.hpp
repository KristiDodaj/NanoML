#pragma once
#include "nanoml/model.hpp"

namespace ml {

class LogisticRegression : public Model {
    public:
        explicit LogisticRegression(std::size_t n_features);

        Vector forward(const Matrix& X) const override;
        void   backward(const Matrix& X,
                        const Vector& dLdy,
                        double lr) override;

    private:
        Vector W;
        double b;
    };
}
