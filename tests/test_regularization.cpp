#include "nanoml/neural_network.hpp"
#include "nanoml/loss.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Simple dataset for testing - XOR problem
void create_xor_data(ml::Matrix& X, ml::Vector& y) {
    X = ml::Matrix(4, 2);
    y = ml::Vector(4);
    
    // Input patterns
    X(0, 0) = 0.0; X(0, 1) = 0.0; // 0 XOR 0 = 0
    X(1, 0) = 0.0; X(1, 1) = 1.0; // 0 XOR 1 = 1
    X(2, 0) = 1.0; X(2, 1) = 0.0; // 1 XOR 0 = 1
    X(3, 0) = 1.0; X(3, 1) = 1.0; // 1 XOR 1 = 0
    
    // Expected outputs
    y[0] = 0.0;
    y[1] = 1.0;
    y[2] = 1.0;
    y[3] = 0.0;
}

// Test L2 regularization
bool test_l2_regularization() {
    ml::Matrix X;
    ml::Vector y;
    create_xor_data(X, y);
    
    // Create a model with regularization
    std::vector<std::size_t> layer_sizes = {2, 4, 1};
    ml::NeuralNetwork model(layer_sizes);
    
    // Set L2 regularization
    model.setL2Regularization(0.01);
    
    // Check if regularization parameter is set correctly
    if (model.getL2Regularization() != 0.01) {
        std::cerr << "L2 regularization parameter not set correctly" << std::endl;
        return false;
    }
    
    // Train for a few iterations
    double learning_rate = 0.5;
    int epochs = 100;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.rows(); ++i) {
            // Create single sample
            ml::Matrix example(1, X.cols());
            for (size_t j = 0; j < X.cols(); ++j) {
                example(0, j) = X(i, j);
            }
            
            // Forward pass
            ml::Vector output = model.forward(example);
            
            // Calculate loss
            double loss = ml::bce(ml::Vector(1, output[0]), ml::Vector(1, y[i]));
            total_loss += loss;
            
            // Compute gradient
            ml::Vector gradient(1);
            gradient[0] = output[0] - y[i];  // Gradient of BCE with sigmoid
            
            // Backward pass with regularization
            model.backward(example, gradient, learning_rate);
        }
    }
    
    // The test passes if we reach this point without errors
    return true;
}

// Test L1 regularization
bool test_l1_regularization() {
    ml::Matrix X;
    ml::Vector y;
    create_xor_data(X, y);
    
    // Create a model with regularization
    std::vector<std::size_t> layer_sizes = {2, 4, 1};
    ml::NeuralNetwork model(layer_sizes);
    
    // Set L1 regularization
    model.setL1Regularization(0.01);
    
    // Check if regularization parameter is set correctly
    if (model.getL1Regularization() != 0.01) {
        std::cerr << "L1 regularization parameter not set correctly" << std::endl;
        return false;
    }
    
    // Train for a few iterations
    double learning_rate = 0.5;
    int epochs = 100;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.rows(); ++i) {
            // Create single sample
            ml::Matrix example(1, X.cols());
            for (size_t j = 0; j < X.cols(); ++j) {
                example(0, j) = X(i, j);
            }
            
            // Forward pass
            ml::Vector output = model.forward(example);
            
            // Calculate loss
            double loss = ml::bce(ml::Vector(1, output[0]), ml::Vector(1, y[i]));
            total_loss += loss;
            
            // Compute gradient
            ml::Vector gradient(1);
            gradient[0] = output[0] - y[i];  // Gradient of BCE with sigmoid
            
            // Backward pass with regularization
            model.backward(example, gradient, learning_rate);
        }
    }
    
    // The test passes if we reach this point without errors
    return true;
}

// Test early stopping
bool test_early_stopping() {
    ml::Matrix X;
    ml::Vector y;
    create_xor_data(X, y);
    
    // Create a model with early stopping
    std::vector<std::size_t> layer_sizes = {2, 4, 1};
    ml::NeuralNetwork model(layer_sizes);
    
    // Enable early stopping with patience = 5
    model.enableEarlyStopping(5, 0.001);
    
    // Check if early stopping parameters are set correctly
    if (!model.isEarlyStoppingEnabled() || model.getEarlyStoppingPatience() != 5) {
        std::cerr << "Early stopping parameters not set correctly" << std::endl;
        return false;
    }
    
    // Reset early stopping counter
    model.resetEarlyStopping();
    
    // Test decreasing loss (shouldn't trigger early stopping)
    for (double loss = 1.0; loss >= 0.5; loss -= 0.1) {
        if (model.shouldStopEarly(loss)) {
            std::cerr << "Early stopping triggered incorrectly for decreasing loss" << std::endl;
            return false;
        }
    }
    
    // Reset early stopping counter
    model.resetEarlyStopping();
    
    // Test with stagnating loss (should trigger early stopping after patience exceeded)
    model.shouldStopEarly(0.5);  // First call, establishes the baseline
    
    // Next 4 calls should not trigger early stopping yet
    for (int i = 0; i < 4; ++i) {
        if (model.shouldStopEarly(0.5)) {
            std::cerr << "Early stopping triggered too early" << std::endl;
            return false;
        }
    }
    
    // The 5th call should trigger early stopping (patience = 5)
    if (!model.shouldStopEarly(0.5)) {
        std::cerr << "Early stopping didn't trigger when it should have" << std::endl;
        return false;
    }
    
    // Test with slightly decreased loss (within min_delta)
    model.resetEarlyStopping();
    model.shouldStopEarly(0.5);  // Baseline
    
    // Loss decreased but not enough to reset patience
    for (int i = 0; i < 4; ++i) {
        if (model.shouldStopEarly(0.4995)) {  // Less than min_delta=0.001 improvement
            std::cerr << "Early stopping triggered despite small improvement" << std::endl;
            return false;
        }
    }
    
    // Test with sufficient decrease (beyond min_delta)
    model.resetEarlyStopping();
    model.shouldStopEarly(0.5);  // Baseline
    
    if (model.shouldStopEarly(0.48)) {  // More than min_delta=0.001 improvement
        std::cerr << "Early stopping triggered despite significant improvement" << std::endl;
        return false;
    }
    
    // Test with disabled early stopping
    model.disableEarlyStopping();
    
    if (model.isEarlyStoppingEnabled() || model.shouldStopEarly(1.0)) {
        std::cerr << "Early stopping still active after being disabled" << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    bool l2_test_passed = test_l2_regularization();
    std::cout << "L2 Regularization Test: " << (l2_test_passed ? "PASSED" : "FAILED") << std::endl;
    
    bool l1_test_passed = test_l1_regularization();
    std::cout << "L1 Regularization Test: " << (l1_test_passed ? "PASSED" : "FAILED") << std::endl;
    
    bool early_stopping_test_passed = test_early_stopping();
    std::cout << "Early Stopping Test: " << (early_stopping_test_passed ? "PASSED" : "FAILED") << std::endl;
    
    return (l2_test_passed && l1_test_passed && early_stopping_test_passed) ? 0 : 1;
}