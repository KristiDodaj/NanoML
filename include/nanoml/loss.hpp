#pragma once
#include "nanoml/vector.hpp"

namespace ml {

    double mse(const Vector& y_hat, const Vector& y);
    Vector mse_grad(const Vector& y_hat, const Vector& y);

}
