#include "nanoml/linear_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>
#include <cmath>

int main() {
    // Tiny dataset: y = 3x + 2  for x = 0..3
    ml::Matrix X(4,1);
    ml::Vector y(4);
    for (int i=0;i<4;++i){ X(i,0)=i; y[i]=3*i+2; }

    ml::LinearRegression model(1);
    ml::GradientDescent  gd({.lr=0.1, .epochs=500, .verbose=false});
    gd.fit(model, X, y, ml::mse, ml::mse_grad);

    auto y_hat = model.forward(X);
    std::cout << "pred for x=3 → " << y_hat[3] << " (target 11)\n";
    return std::fabs(y_hat[3]-11.0) < 0.2 ? 0 : 1;
}
