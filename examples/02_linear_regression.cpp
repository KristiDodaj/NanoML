#include "nanoml/linear_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <iostream>
#include <iomanip>

/**
 * @file 02_linear_regression.cpp
 * @brief Example demonstrating linear regression using nanoml
 */
int main() {
    std::cout << "NanoML Linear Regression Example\n";
    std::cout << "===============================\n\n";
    
    // Create synthetic dataset: y = 2x + 1 + noise
    ml::Matrix X(5, 1);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 4.0;
    X(4, 0) = 5.0;
    
    ml::Vector y(5);
    y[0] = 2.9;  // 2*1 + 1 + noise
    y[1] = 5.1;  // 2*2 + 1 + noise
    y[2] = 7.2;  // 2*3 + 1 + noise
    y[3] = 9.0;  // 2*4 + 1 + noise
    y[4] = 10.8; // 2*5 + 1 + noise
    
    std::cout << "Training data:\n";
    std::cout << "X = [";
    for (size_t i = 0; i < X.rows(); ++i) {
        std::cout << X(i, 0);
        if (i < X.rows() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "y = [";
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << y[i];
        if (i < y.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
    
    // Create and train linear regression model
    ml::LinearRegression model(1);  // 1 feature
    
    // Configure gradient descent optimizer
    ml::GDConfig config;
    config.lr = 0.01;              // learning rate
    config.epochs = 1000;          // number of epochs
    config.verbose = true;         // print progress
    config.log_interval = 200;     // log every 200 epochs
    
    ml::GradientDescent optimizer(config);
    
    std::cout << "Training linear regression model...\n";
    optimizer.fit(model, X, y, ml::mse, ml::mse_grad);
    
    // For demonstration, let's manually create a test input to see predictions
    std::cout << "\nModel trained. Let's make some predictions.\n";
    
    // Make predictions for new data
    std::cout << "Making predictions:\n";
    for (double x = 0.0; x <= 6.0; x += 1.0) {
        ml::Matrix x_test(1, 1);
        x_test(0, 0) = x;
        ml::Vector y_pred = model.forward(x_test);
        std::cout << "x = " << x << ", predicted y = " << std::fixed << std::setprecision(4) << y_pred[0] << "\n";
    }
    
    return 0;
}