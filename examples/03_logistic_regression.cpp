#include "nanoml/logistic_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/metrics.hpp"
#include <iostream>
#include <iomanip>

/**
 * @file 03_logistic_regression.cpp
 * @brief Example demonstrating logistic regression for binary classification using nanoml
 */
int main() {
    std::cout << "NanoML Logistic Regression Example\n";
    std::cout << "=================================\n\n";
    
    // Create a binary classification dataset
    ml::Matrix X(6, 2);
    // Feature 1
    X(0, 0) = 1.0;  X(1, 0) = 2.0;  X(2, 0) = 1.5;
    X(3, 0) = 6.0;  X(4, 0) = 7.0;  X(5, 0) = 5.5;
    // Feature 2
    X(0, 1) = 1.1;  X(1, 1) = 1.3;  X(2, 1) = 2.0;
    X(3, 1) = 5.8;  X(4, 1) = 6.5;  X(5, 1) = 7.0;
    
    // Binary labels: 0 or 1
    ml::Vector y(6);
    y[0] = 0.0;  y[1] = 0.0;  y[2] = 0.0;
    y[3] = 1.0;  y[4] = 1.0;  y[5] = 1.0;
    
    std::cout << "Training data:\n";
    for (size_t i = 0; i < X.rows(); ++i) {
        std::cout << "X[" << i << "] = [" << X(i, 0) << ", " << X(i, 1) << "], y = " << y[i] << "\n";
    }
    std::cout << "\n";
    
    // Create and train logistic regression model
    ml::LogisticRegression model(2); // 2 features
    
    // Configure gradient descent optimizer
    ml::GDConfig config;
    config.lr = 0.1;              // learning rate
    config.epochs = 1000;         // number of epochs
    config.verbose = true;        // print progress
    config.log_interval = 200;    // log every 200 epochs
    
    ml::GradientDescent optimizer(config);
    
    std::cout << "Training logistic regression model...\n";
    optimizer.fit(model, X, y, ml::bce, ml::bce_grad);
    
    std::cout << "\nModel trained. Let's evaluate the predictions.\n";
    
    // Make predictions on training data
    ml::Vector y_probs = model.forward(X);
    ml::Vector y_pred(y_probs.size());
    
    std::cout << "Predictions on training data:\n";
    for (size_t i = 0; i < X.rows(); ++i) {
        // Convert probability to binary prediction (threshold at 0.5)
        y_pred[i] = (y_probs[i] > 0.5) ? 1.0 : 0.0;
        std::cout << "Sample " << i << ": features = [" << X(i, 0) << ", " << X(i, 1) 
                  << "], true class = " << y[i] 
                  << ", probability = " << std::fixed << std::setprecision(4) << y_probs[i] 
                  << ", predicted class = " << y_pred[i] << "\n";
    }
    
    // Calculate accuracy
    double accuracy = ml::accuracy(y_pred, y);
    std::cout << "\nModel accuracy: " << std::fixed << std::setprecision(4) << (accuracy * 100) << "%\n";
    
    return 0;
}