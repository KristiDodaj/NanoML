#include "nanoml/neural_network.hpp"
#include "nanoml/activations.hpp"
#include "nanoml/loss.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/metrics.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

/**
 * @file 04_neural_network.cpp
 * @brief Example demonstrating neural network implementation with nanoml
 *        Uses an XOR problem as demonstration
 */
int main() {
    std::cout << "NanoML Neural Network Example: XOR Problem\n";
    std::cout << "========================================\n\n";
    
    // XOR problem dataset
    ml::Matrix X(4, 2);  // 4 examples, 2 features each
    X(0, 0) = 0.0; X(0, 1) = 0.0;  // Input [0,0]
    X(1, 0) = 0.0; X(1, 1) = 1.0;  // Input [0,1]
    X(2, 0) = 1.0; X(2, 1) = 0.0;  // Input [1,0]
    X(3, 0) = 1.0; X(3, 1) = 1.0;  // Input [1,1]
    
    ml::Vector y(4);  // XOR outputs
    y[0] = 0.0;  // 0 XOR 0 = 0
    y[1] = 1.0;  // 0 XOR 1 = 1
    y[2] = 1.0;  // 1 XOR 0 = 1
    y[3] = 0.0;  // 1 XOR 1 = 0
    
    std::cout << "XOR Truth Table (Training Data):\n";
    std::cout << "x1\tx2\ty\n";
    std::cout << "-------------------\n";
    for (size_t i = 0; i < X.rows(); ++i) {
        std::cout << X(i, 0) << "\t" << X(i, 1) << "\t" << y[i] << "\n";
    }
    std::cout << "\n";
    
    // Create a neural network with 2 inputs, a hidden layer with 4 neurons, and 1 output
    // Using the vector constructor which defines the neurons per layer
    std::vector<std::size_t> layer_sizes = {2, 4, 1};
    ml::NeuralNetwork network(layer_sizes);
    
    std::cout << "Neural Network Architecture:\n";
    std::cout << "Input: " << layer_sizes[0] << " neurons\n";
    std::cout << "Hidden: " << layer_sizes[1] << " neurons (Sigmoid activation)\n";
    std::cout << "Output: " << layer_sizes[2] << " neuron (Sigmoid activation)\n\n";
    
    // Train the neural network manually to avoid the assertion error
    std::cout << "Training neural network...\n";
    double learning_rate = 0.5;
    size_t epochs = 10000;
    std::cout << "[INFO] Starting training with " << epochs << " epochs\n";
    std::cout << "[INFO] Learning rate: " << std::fixed << std::setprecision(6) << learning_rate << "\n";
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        // Train on each example
        for (size_t i = 0; i < X.rows(); ++i) {
            // Create input for a single sample
            ml::Matrix input_row(1, X.cols());
            input_row(0, 0) = X(i, 0);
            input_row(0, 1) = X(i, 1);
            
            // Forward pass
            ml::Vector output = network.forward(input_row);
            
            // Create target with the same size as the output
            ml::Vector target(output.size());
            target[0] = y[i];
            
            // Compute loss for this sample
            double sample_loss = ml::bce(output, target);
            total_loss += sample_loss;
            
            // Compute gradient and update weights
            ml::Vector grad = ml::bce_grad(output, target);
            network.backward(input_row, grad, learning_rate);
        }
        
        // Log training progress
        total_loss /= X.rows();
        if (epoch % 1000 == 0 || epoch == epochs - 1) {
            std::cout << "[INFO] epoch " << epoch << ", loss = " << std::fixed << std::setprecision(6) << total_loss << "\n";
        }
    }
    
    // Evaluate the trained model
    std::cout << "\nEvaluation on training data:\n";
    std::cout << "x1\tx2\tTrue\tPredicted\tRounded\n";
    std::cout << "-------------------------------------------\n";
    
    ml::Vector predictions(y.size());
    for (size_t i = 0; i < X.rows(); ++i) {
        ml::Matrix input_row(1, X.cols());
        input_row(0, 0) = X(i, 0);
        input_row(0, 1) = X(i, 1);
        
        ml::Vector output = network.forward(input_row);
        double predicted = output[0];
        double rounded = (predicted > 0.5) ? 1.0 : 0.0;
        predictions[i] = rounded;
        
        std::cout << X(i, 0) << "\t" << X(i, 1) << "\t" 
                 << y[i] << "\t" << std::fixed << std::setprecision(4) << predicted 
                 << "\t\t" << rounded << "\n";
    }
    
    // Calculate accuracy
    double accuracy = ml::accuracy(predictions, y);
    std::cout << "\nModel accuracy: " << std::fixed << std::setprecision(4) << (accuracy * 100) << "%\n";
    
    return 0;
}