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
                        
        bool save(const std::string& filename) const override;
        bool load(const std::string& filename) override;
        std::string get_model_type() const override { return "LogisticRegression"; }

    private:
        Vector W;
        double b;
    };
}
