#include "nanoml/linear_regression.hpp"
#include "nanoml/loss.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <filesystem>

// Helper function for approximate comparison of doubles
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

// Helper function for approximate comparison of vectors
bool vectors_approx_equal(const ml::Vector& a, const ml::Vector& b, double epsilon = 1e-9) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], epsilon)) return false;
    }
    return true;
}

int main() {
    std::cout << "=== Testing Model Serialization ===" << std::endl;
    
    // Create a simple dataset: y = 2*x + 1
    ml::Matrix X(5, 1);
    ml::Vector y(5);
    
    // Set up training data
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 4.0;
    X(4, 0) = 5.0;
    
    y[0] = 3.0;   // 2*1 + 1
    y[1] = 5.0;   // 2*2 + 1
    y[2] = 7.0;   // 2*3 + 1
    y[3] = 9.0;   // 2*4 + 1
    y[4] = 11.0;  // 2*5 + 1
    
    // Create and train a linear regression model
    ml::LinearRegression model(1);  // 1 feature
    
    // Manual training loop
    double learning_rate = 0.01;
    int epochs = 1000;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        ml::Vector predictions = model.forward(X);
        ml::Vector gradients = ml::mse_grad(predictions, y);
        model.backward(X, gradients, learning_rate);
    }
    
    // Predictions from the trained model
    ml::Vector original_predictions = model.forward(X);
    
    std::cout << "Original model trained. Saving to file..." << std::endl;
    
    // Save the model to file
    const std::string model_file = "test_model.bin";
    bool save_success = model.save(model_file);
    assert(save_success && "Failed to save model");
    
    // Create a new model and load from file
    ml::LinearRegression loaded_model(1);
    std::cout << "Loading model from file..." << std::endl;
    bool load_success = loaded_model.load(model_file);
    assert(load_success && "Failed to load model");
    
    // Get predictions from loaded model
    ml::Vector loaded_predictions = loaded_model.forward(X);
    
    // Compare predictions
    std::cout << "Comparing predictions..." << std::endl;
    assert(vectors_approx_equal(original_predictions, loaded_predictions) && 
           "Predictions from loaded model don't match original model");
    
    // Test with new data point
    ml::Matrix new_X(1, 1);
    new_X(0, 0) = 10.0; // Should predict close to 21.0 (2*10 + 1)
    
    double original_pred = model.forward(new_X)[0];
    double loaded_pred = loaded_model.forward(new_X)[0];
    
    std::cout << "Original model prediction for x=10: " << original_pred << std::endl;
    std::cout << "Loaded model prediction for x=10: " << loaded_pred << std::endl;
    
    assert(approx_equal(original_pred, loaded_pred) && 
           "Predictions for new data don't match between models");
    
    // Clean up the test file
    std::filesystem::remove(model_file);
    
    std::cout << "\nAll model serialization tests passed ✅\n";
    return 0;
}