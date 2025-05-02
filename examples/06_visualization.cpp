#include "nanoml/visualization.hpp"
#include "nanoml/logger.hpp"
#include "nanoml/linear_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>

/**
 * @file 06_visualization.cpp
 * @brief Example demonstrating visualization capabilities in nanoml
 */

// Function to normalize features (standardization: z-score normalization)
void normalize_features(ml::Matrix& X, double& mean, double& std_dev) {
    // Calculate mean
    mean = 0.0;
    for (int i = 0; i < X.rows(); ++i) {
        mean += X(i, 0);
    }
    mean /= X.rows();
    
    // Calculate standard deviation
    std_dev = 0.0;
    for (int i = 0; i < X.rows(); ++i) {
        std_dev += (X(i, 0) - mean) * (X(i, 0) - mean);
    }
    std_dev = std::sqrt(std_dev / X.rows());
    
    // Avoid division by zero
    if (std_dev < 1e-10) {
        std_dev = 1.0;
    }
    
    // Normalize the data
    for (int i = 0; i < X.rows(); ++i) {
        X(i, 0) = (X(i, 0) - mean) / std_dev;
    }
}

int main() {
    std::cout << "NanoML Visualization Example\n";
    std::cout << "===========================\n\n";
    
    // Create a simple dataset for a linear regression problem
    std::cout << "Generating synthetic data...\n";
    const int n_samples = 20;
    ml::Matrix X(n_samples, 1);
    ml::Vector y(n_samples);
    
    // Generate data with the equation: y = 2x + 1 + noise
    std::srand(42); // For reproducibility
    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = i * 0.5;  // x values from 0 to 9.5
        
        // Add some random noise
        double noise = (std::rand() % 1000 - 500) / 500.0;  // Random noise between -1 and 1
        y[i] = 2.0 * X(i, 0) + 1.0 + noise;
    }
    
    std::cout << "Data generated with equation: y = 2x + 1 + noise\n\n";
    
    // Normalize the features to prevent numerical issues
    double X_mean = 0.0, X_std = 1.0;
    normalize_features(X, X_mean, X_std);
    std::cout << "Features normalized (mean=" << X_mean << ", std=" << X_std << ")\n\n";
    
    // Create and train a linear regression model
    std::cout << "Training linear regression model...\n";
    ml::LinearRegression model(1);  // 1 feature
    
    // Configure an optimizer with logging enabled
    ml::GDConfig config;
    config.lr = 0.01;          // reduced learning rate (was 0.1)
    config.epochs = 100;       // number of epochs
    config.verbose = true;     // print progress
    config.log_interval = 10;  // log every 10 epochs
    
    ml::GradientDescent optimizer(config);
    
    // Train the model (this will also log the training metrics)
    optimizer.fit(model, X, y, ml::mse, ml::mse_grad);
    
    // Get the logger from the optimizer
    auto logger = optimizer.get_logger();
    
    // Create a visualizer
    ml::Visualizer viz;
    
    // 1. ASCII Plot
    std::cout << "\nASCII Visualization of Training Loss:\n";
    viz.plot_metrics_ascii(*logger, "loss");
    
    // 2. Export metrics to CSV
    std::string csv_filename = "training_metrics.csv";
    bool csv_success = viz.export_metrics_csv(*logger, csv_filename);
    if (csv_success) {
        std::cout << "\nTraining metrics exported to CSV: " << csv_filename << "\n";
        std::cout << "You can use this file to visualize the training metrics in external tools.\n";
    } else {
        std::cout << "\nFailed to export training metrics to CSV.\n";
    }
    
    // 3. Export metrics to HTML for interactive visualization
    std::string html_filename = "training_visualization.html";
    bool html_success = viz.export_metrics_html(*logger, html_filename);
    if (html_success) {
        std::cout << "\nTraining visualization exported to HTML: " << html_filename << "\n";
        std::cout << "Open this file in a web browser to view the interactive chart.\n";
        
        // Print the absolute path for convenience
        std::filesystem::path abs_path = std::filesystem::absolute(html_filename);
        std::cout << "Full path: " << abs_path.string() << "\n";
    } else {
        std::cout << "\nFailed to export visualization to HTML.\n";
    }
    
    // Make predictions for visualization
    std::cout << "\nPredictions for visualization:\n";
    std::cout << "x\tActual y\tPredicted y\n";
    std::cout << "--------------------------------\n";
    
    // Display a few data points and predictions
    for (int i = 0; i < n_samples; i += 4) {
        // Original x value (unnormalized)
        double orig_x = i * 0.5;
        
        // Normalized x value for prediction
        ml::Matrix x_point(1, 1);
        x_point(0, 0) = (orig_x - X_mean) / X_std;
        
        ml::Vector y_pred = model.forward(x_point);
        
        std::cout << std::fixed << std::setprecision(2) 
                  << orig_x << "\t" 
                  << y[i] << "\t\t" 
                  << y_pred[0] << "\n";
    }
    
    return 0;
}