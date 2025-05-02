#include "nanoml/csv.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/linear_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

/**
 * @file 05_csv_data_handling.cpp
 * @brief Example demonstrating CSV data loading and processing in nanoml
 */

// Function to generate a sample CSV file for housing data
void generate_housing_data_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "size_sqft,bedrooms,age,price\n";
    
    // Generate data points with a formula: price = 100 * size + 5000 * bedrooms - 2000 * age + 50000 + noise
    std::vector<std::tuple<double, double, double, double>> data_points = {
        {1000, 1, 15, 100*1000 + 5000*1 - 2000*15 + 50000 + 2000},  // with some deterministic "noise"
        {1500, 2, 10, 100*1500 + 5000*2 - 2000*10 + 50000 - 3000},
        {2000, 3, 5, 100*2000 + 5000*3 - 2000*5 + 50000 + 1500},
        {2500, 4, 3, 100*2500 + 5000*4 - 2000*3 + 50000 - 2000},
        {1800, 2, 8, 100*1800 + 5000*2 - 2000*8 + 50000 + 5000},
        {3000, 5, 1, 100*3000 + 5000*5 - 2000*1 + 50000 + 0},
        {1200, 1, 12, 100*1200 + 5000*1 - 2000*12 + 50000 - 4000},
        {1600, 2, 9, 100*1600 + 5000*2 - 2000*9 + 50000 + 3500},
        {2200, 3, 4, 100*2200 + 5000*3 - 2000*4 + 50000 - 1000},
        {2800, 4, 2, 100*2800 + 5000*4 - 2000*2 + 50000 + 2500}
    };
    
    // Write data to file
    for (const auto& [size, bedrooms, age, price] : data_points) {
        file << size << "," << bedrooms << "," << age << "," << static_cast<int>(price) << "\n";
    }
    
    file.close();
    std::cout << "Generated housing data CSV file: " << filename << "\n";
}

int main() {
    std::cout << "NanoML CSV Data Handling Example\n";
    std::cout << "==============================\n\n";
    
    // Create a sample CSV file
    std::string csv_path = "housing_data.csv";
    generate_housing_data_csv(csv_path);
    
    // Load the CSV file
    std::cout << "Loading data from CSV file...\n";
    ml::CSVReader reader(csv_path, true);  // true means the first row is header
    
    // Get basic info about the data
    std::cout << "\nCSV File Summary:\n";
    std::cout << "Number of rows: " << reader.rows() << "\n";
    std::cout << "Number of columns: " << reader.cols() << "\n";
    std::cout << "Headers: ";
    const auto& headers = reader.headers();
    for (size_t i = 0; i < headers.size(); ++i) {
        std::cout << headers[i];
        if (i < headers.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";
    
    // Extract features and target
    std::vector<size_t> feature_cols = {0, 1, 2};  // size_sqft, bedrooms, age
    ml::Matrix X = reader.get_columns(feature_cols);
    ml::Vector y = reader.get_column(3);  // price
    
    std::cout << "First 3 rows of data:\n";
    std::cout << "size_sqft\tbedrooms\tage\tprice\n";
    std::cout << "--------------------------------------\n";
    for (size_t i = 0; i < 3 && i < X.rows(); ++i) {
        std::cout << X(i, 0) << "\t\t" << X(i, 1) << "\t\t" << X(i, 2) << "\t" << y[i] << "\n";
    }
    std::cout << "...\n\n";
    
    // Perform simple data normalization (standardization)
    std::cout << "Normalizing features (standardization)...\n";
    for (size_t j = 0; j < X.cols(); ++j) {
        double mean = 0.0;
        double std_dev = 0.0;
        
        // Calculate mean
        for (size_t i = 0; i < X.rows(); ++i) {
            mean += X(i, j);
        }
        mean /= X.rows();
        
        // Calculate standard deviation
        for (size_t i = 0; i < X.rows(); ++i) {
            std_dev += (X(i, j) - mean) * (X(i, j) - mean);
        }
        std_dev = std::sqrt(std_dev / X.rows());
        
        // Standardize the feature
        for (size_t i = 0; i < X.rows(); ++i) {
            X(i, j) = (X(i, j) - mean) / std_dev;
        }
        
        std::cout << "Feature " << j << " (";
        if (j < headers.size() - 1) {
            std::cout << headers[j];
        }
        std::cout << "): mean=" << mean << ", std_dev=" << std_dev << "\n";
    }
    std::cout << "\n";
    
    // Train a linear regression model
    std::cout << "Training linear regression model on the data...\n";
    ml::LinearRegression model(X.cols());
    
    ml::GDConfig config;
    config.lr = 0.01;           // learning rate
    config.epochs = 1000;       // number of epochs
    config.verbose = true;      // print progress
    config.log_interval = 200;  // log every 200 epochs
    
    ml::GradientDescent optimizer(config);
    
    optimizer.fit(model, X, y, ml::mse, ml::mse_grad);
    
    // Make predictions on the training data
    ml::Vector predictions = model.forward(X);
    
    // Calculate R-squared
    double y_mean = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        y_mean += y[i];
    }
    y_mean /= y.size();
    
    double ss_total = 0.0;
    double ss_residual = 0.0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        ss_total += (y[i] - y_mean) * (y[i] - y_mean);
        ss_residual += (y[i] - predictions[i]) * (y[i] - predictions[i]);
    }
    
    double r_squared = 1.0 - (ss_residual / ss_total);
    
    std::cout << "\nModel evaluation:\n";
    std::cout << "R-squared: " << std::fixed << std::setprecision(4) << r_squared << "\n";
    std::cout << "(A value close to 1.0 indicates a good fit)\n";
    
    return 0;
}