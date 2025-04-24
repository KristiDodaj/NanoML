#pragma once
#include "nanoml/model.hpp"
#include <iostream>

namespace ml {
    struct GDConfig {
        double       lr     = 0.01;
        std::size_t  epochs = 1000;
        bool         verbose = false;
};

class GradientDescent {
    public:
        explicit GradientDescent(GDConfig cfg = {}) : cfg_(cfg) {}

        template<typename Loss, typename LossGrad>
        void fit(Model&       model,
                const Matrix& X,
                const Vector& y,
                Loss         loss_fn,
                LossGrad     grad_fn)
        {
            for (std::size_t e = 0; e < cfg_.epochs; ++e) {
                Vector y_hat = model.forward(X);
                double L     = loss_fn(y_hat, y);
                model.backward(X, grad_fn(y_hat, y), cfg_.lr);

                if (cfg_.verbose && e % 100 == 0)
                    std::cout << "epoch " << e << "  loss = " << L << '\n';
            }
        }
    private:
        GDConfig cfg_;
    };

}
