# 🧠 NanoML: Lightweight Machine Learning Library in C++

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/nanoml)
[![Version](https://img.shields.io/badge/version-1.0.0-orange)](https://github.com/yourusername/nanoml/releases)

</div>

> **NanoML** is a lightweight machine learning library built from scratch in modern C++ (C++20). It provides a clean, easy-to-use API for common machine learning tasks with a focus on educational value and flexibility for small to medium-sized datasets.

## 📚 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Working with CSV Files](#-working-with-csv-files)
- [Data Preprocessing](#-data-preprocessing)
- [Evaluation Metrics](#-evaluation-metrics)
- [Cross-Validation](#-cross-validation)
- [Model Serialization](#-model-serialization)
- [Visualization](#-visualization)
- [Building Neural Networks](#-building-neural-networks-with-layers)
- [Advanced Features](#-advanced-neural-network-features)
- [License](#-license)

## ✨ Features

<table>
  <tr>
    <td>
      <h4>📊 Core Data Structures</h4>
      <ul>
        <li>Vector and Matrix classes with comprehensive mathematical operations</li>
        <li>CSV file reading and data loading utilities</li>
      </ul>
    </td>
    <td>
      <h4>🔮 Machine Learning Models</h4>
      <ul>
        <li>Linear Regression</li>
        <li>Logistic Regression</li>
        <li>Neural Networks</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h4>🧠 Neural Network Components</h4>
      <ul>
        <li>Dense (fully connected) layers</li>
        <li>Activation layers (ReLU, Sigmoid, Tanh)</li>
        <li>Batch Normalization</li>
        <li>Dropout regularization</li>
        <li>Convolutional layers</li>
        <li>Max Pooling layers</li>
        <li>Flatten layers</li>
      </ul>
    </td>
    <td>
      <h4>⚙️ Training & Optimization</h4>
      <ul>
        <li>Gradient Descent optimizer</li>
        <li>Mini-batch processing</li>
        <li>Early stopping</li>
        <li>L1/L2 regularization</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h4>📉 Loss Functions</h4>
      <ul>
        <li>Mean Squared Error (MSE)</li>
        <li>Binary Cross-Entropy (BCE)</li>
      </ul>
    </td>
    <td>
      <h4>📏 Validation & Metrics</h4>
      <ul>
        <li>K-fold cross-validation</li>
        <li>Train-test splitting</li>
        <li>R-squared, MSE, accuracy metrics</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h4>📊 Visualization</h4>
      <ul>
        <li>ASCII plot generation in terminal</li>
        <li>Export metrics to CSV</li>
        <li>Interactive HTML visualization with Chart.js</li>
      </ul>
    </td>
    <td>
      <h4>💾 Model Serialization</h4>
      <ul>
        <li>Save and load trained models</li>
      </ul>
    </td>
  </tr>
</table>

## 🚀 Installation

### Prerequisites

- CMake 3.20+
- C++20 compatible compiler:
  - GCC 10+
  - Clang 11+
  - MSVC 19.28+

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/nanoml.git
cd nanoml

# Create a build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run tests to verify installation
make test
```

## 🏃‍♂️ Quick Start

<details>
<summary><b>Linear Regression Example</b> (click to expand)</summary>

```cpp
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/linear_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>

int main() {
    // Create a simple dataset: y = 2x + 1
    ml::Matrix X(5, 1);
    ml::Vector y(5);
    
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
    
    ml::GDConfig config {
        .lr = 0.01,            // Learning rate
        .epochs = 1000,        // Number of training epochs
        .verbose = true,       // Show training progress
        .log_interval = 100    // Log every 100 epochs
    };
    
    ml::GradientDescent optimizer(config);
    optimizer.fit(model, X, y, ml::mse, ml::mse_grad);
    
    // Make predictions
    ml::Matrix newX(1, 1);
    newX(0, 0) = 6.0;
    ml::Vector pred = model.forward(newX);
    
    std::cout << "Prediction for x=6: " << pred[0] << std::endl;
    
    // Save the model
    model.save("linear_model.bin");
    
    return 0;
}
```
</details>

<details>
<summary><b>Logistic Regression Example</b> (click to expand)</summary>

```cpp
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/logistic_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>

int main() {
    // Binary classification dataset (OR gate)
    ml::Matrix X(4, 2);
    ml::Vector y(4);
    
    X(0, 0) = 0.0; X(0, 1) = 0.0; y[0] = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0; y[1] = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 0.0; y[2] = 1.0;
    X(3, 0) = 1.0; X(3, 1) = 1.0; y[3] = 1.0;
    
    ml::LogisticRegression model(2);  // 2 features
    
    ml::GDConfig config {
        .lr = 0.5,            // Learning rate
        .epochs = 2000,       // Number of training epochs
        .verbose = true       // Show training progress
    };
    
    ml::GradientDescent optimizer(config);
    optimizer.fit(model, X, y, ml::bce, ml::bce_grad);
    
    // Make predictions
    ml::Vector predictions = model.forward(X);
    
    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << X(i, 0) << " OR " << X(i, 1) << " = " << predictions[i] << std::endl;
    }
    
    return 0;
}
```
</details>

<details>
<summary><b>Neural Network Example</b> (click to expand)</summary>

```cpp
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/neural_network.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // XOR dataset
    ml::Matrix X(4, 2);
    ml::Vector y(4);
    
    X(0, 0) = 0.0; X(0, 1) = 0.0; y[0] = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0; y[1] = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 0.0; y[2] = 1.0;
    X(3, 0) = 1.0; X(3, 1) = 1.0; y[3] = 0.0;
    
    // Define network architecture
    std::vector<std::size_t> layer_sizes = {2, 4, 1};  // 2 inputs, 4 hidden, 1 output
    ml::NeuralNetwork network(layer_sizes);
    
    // Configure and train with GradientDescent
    ml::GDConfig config{
        .lr = 0.5,
        .epochs = 10000,
        .verbose = true,
        .log_interval = 1000
    };
    ml::GradientDescent optimizer(config);
    optimizer.fit(network, X, y, ml::bce, ml::bce_grad);
    
    // Test the trained model
    std::cout << "Predictions:" << std::endl;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        ml::Matrix sample(1, X.cols());
        for (std::size_t j = 0; j < X.cols(); ++j) {
            sample(0, j) = X(i, j);
        }
        
        ml::Vector output = network.forward(sample);
        std::cout << X(i, 0) << " XOR " << X(i, 1) << " = " << output[0] << std::endl;
    }
    
    return 0;
}
```
</details>

## 📊 Working with CSV Files

```cpp
#include "nanoml/csv.hpp"
#include "nanoml/linear_regression.hpp"
#include <iostream>
#include <string>

int main() {
    // Load CSV data
    ml::CSVReader reader("housing_data.csv");
    
    if (!reader.is_loaded()) {
        std::cerr << "Failed to load CSV file" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << reader.rows() << " rows and " 
              << reader.cols() << " columns" << std::endl;
    
    // Extract features using column names
    ml::Matrix X = reader.get_columns({"size_sqft", "bedrooms", "age"});
    
    // Extract target variable
    ml::Vector y = reader.get_column("price");
    
    // Use the data for machine learning
    // ...
    
    return 0;
}
```

## 🧹 Data Preprocessing

<details>
<summary><b>Normalization</b> (click to expand)</summary>

```cpp
#include "nanoml/matrix.hpp"
#include <cmath>

// Using the library's built-in matrix operations for normalization
// The code below demonstrates the standardization process (zero mean, unit variance)
void normalize_features(ml::Matrix& X) {
    for (size_t col = 0; col < X.cols(); ++col) {
        // Calculate mean
        double mean = 0.0;
        for (size_t row = 0; row < X.rows(); ++row) {
            mean += X(row, col);
        }
        mean /= X.rows();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (size_t row = 0; row < X.rows(); ++row) {
            double diff = X(row, col) - mean;
            variance += diff * diff;
        }
        variance /= X.rows();
        double std_dev = std::sqrt(variance);
        
        // Normalize the column
        for (size_t row = 0; row < X.rows(); ++row) {
            X(row, col) = (X(row, col) - mean) / std_dev;
        }
    }
}
```
</details>

<details>
<summary><b>Train-Test Split</b> (click to expand)</summary>

```cpp
#include "nanoml/validation.hpp"
#include <iostream>

int main() {
    // Create sample data
    ml::Matrix X(100, 3); // 100 samples, 3 features
    ml::Vector y(100);    // 100 target values
    
    // Fill X and y with your data...
    
    // Split data into training and test sets (80% train, 20% test)
    ml::Matrix X_train, X_test;
    ml::Vector y_train, y_test;
    
    // Using the library's built-in train_test_split function
    ml::train_test_split(X, y, X_train, y_train, X_test, y_test, 0.2);
    
    std::cout << "Training set: " << X_train.rows() << " samples" << std::endl;
    std::cout << "Test set: " << X_test.rows() << " samples" << std::endl;
    
    // Now you can train on X_train, y_train and evaluate on X_test, y_test
    
    return 0;
}
```
</details>

## 📏 Evaluation Metrics

```cpp
#include "nanoml/metrics.hpp"
#include <iostream>
#include <cmath>

// Example using metrics
void evaluate_model(const ml::Vector& y_true, const ml::Vector& y_pred) {
    // For regression
    double mse = ml::mean_squared_error(y_pred, y_true);
    double rmse = std::sqrt(mse);
    double r2 = ml::r_squared(y_pred, y_true);
    
    std::cout << "Regression metrics:\n";
    std::cout << "  MSE:  " << mse << "\n";
    std::cout << "  RMSE: " << rmse << "\n";
    std::cout << "  R²:   " << r2 << "\n";
    
    // For classification
    double acc = ml::accuracy(y_pred, y_true);
    double precision = ml::precision(y_pred, y_true);
    double recall = ml::recall(y_pred, y_true);
    double f1 = ml::f1_score(y_pred, y_true);
    
    std::cout << "Classification metrics:\n";
    std::cout << "  Accuracy: " << acc << "\n";
    std::cout << "  Precision: " << precision << "\n";
    std::cout << "  Recall: " << recall << "\n";
    std::cout << "  F1 Score: " << f1 << "\n";
}
```

## 🔄 Cross-Validation

```cpp
#include "nanoml/validation.hpp"
#include "nanoml/linear_regression.hpp"
#include "nanoml/loss.hpp"
#include <iostream>

int main() {
    // Prepare your data
    ml::Matrix X = /* ... */;
    ml::Vector y = /* ... */;
    
    // Perform 5-fold cross-validation for regression
    auto cv_scores = ml::cross_validate_regression<ml::LinearRegression, 
                                                  decltype(ml::mse), 
                                                  decltype(ml::mse_grad)>
                    (X, y, 5, 0.01, 1000);
    
    // Print scores
    std::cout << "Cross-validation R² scores:" << std::endl;
    for (size_t i = 0; i < cv_scores.size(); ++i) {
        std::cout << "Fold " << (i+1) << ": " << cv_scores[i] << std::endl;
    }
    
    // Calculate mean score
    double mean_score = 0.0;
    for (double score : cv_scores) {
        mean_score += score;
    }
    mean_score /= cv_scores.size();
    
    std::cout << "Mean R²: " << mean_score << std::endl;
    
    return 0;
}
```

## 💾 Model Serialization

<details>
<summary><b>Saving a Model</b> (click to expand)</summary>

```cpp
// Train your model
ml::LinearRegression model(n_features);
// ... training code ...

// Save the model
std::string filename = "model.bin";
if (model.save(filename)) {
    std::cout << "Model saved successfully to " << filename << std::endl;
} else {
    std::cerr << "Failed to save model" << std::endl;
}
```
</details>

<details>
<summary><b>Loading a Model</b> (click to expand)</summary>

```cpp
// Create a model instance
ml::LinearRegression model(n_features);

// Load the model
std::string filename = "model.bin";
if (model.load(filename)) {
    std::cout << "Model loaded successfully from " << filename << std::endl;
    
    // Use the loaded model for predictions
    ml::Vector predictions = model.forward(X_test);
} else {
    std::cerr << "Failed to load model" << std::endl;
}
```
</details>

## 📈 Visualization

<details>
<summary><b>ASCII Plot in Terminal</b> (click to expand)</summary>

```cpp
#include "nanoml/visualization.hpp"

// After training with gradient descent
auto logger = optimizer.get_logger();

// Create a visualizer
ml::Visualizer viz;

// Generate ASCII plot of loss values
viz.plot_metrics_ascii(*logger, "loss");
```
</details>

<details>
<summary><b>Interactive HTML Visualization</b> (click to expand)</summary>

```cpp
#include "nanoml/visualization.hpp"

// After training
auto logger = optimizer.get_logger();
ml::Visualizer viz;

// Export metrics to HTML with Chart.js
std::string html_file = "training_progress.html";
if (viz.export_metrics_html(*logger, html_file)) {
    std::cout << "Visualization exported to " << html_file << std::endl;
}
```
</details>

## 🧱 Building Neural Networks with Layers

```cpp
#include "nanoml/neural_network.hpp"
#include <vector>

// Create a neural network manually with layers
int main() {
    // Input layer -> Hidden layer with ReLU -> Output layer with Sigmoid
    std::vector<std::unique_ptr<ml::Layer>> layers;
    
    // Input size is 2
    layers.push_back(std::make_unique<ml::DenseLayer>(2, 4));
    layers.push_back(std::make_unique<ml::ReLULayer>());
    layers.push_back(std::make_unique<ml::DenseLayer>(4, 1));
    layers.push_back(std::make_unique<ml::SigmoidLayer>());
    
    // Create a neural network
    ml::NeuralNetwork network(std::move(layers));
    
    // Use the network
    // ...
    
    return 0;
}
```

## 🚀 Advanced Neural Network Features

<details>
<summary><b>Regularization</b> (click to expand)</summary>

```cpp
// Create a neural network
ml::NeuralNetwork network({2, 10, 5, 1});

// Add L1 regularization (lasso)
network.setL1Regularization(0.001);

// Add L2 regularization (ridge)
network.setL2Regularization(0.001);
```
</details>

<details>
<summary><b>Early Stopping</b> (click to expand)</summary>

```cpp
// Enable early stopping (patience = 10, min delta = 0.001)
network.enableEarlyStopping(10, 0.001);

// During training loop
for (int epoch = 0; epoch < max_epochs; ++epoch) {
    // ... training code ...
    
    // Evaluate model on validation set
    double val_loss = evaluate_on_validation_set();
    
    // Check for early stopping
    if (network.shouldStopEarly(val_loss)) {
        std::cout << "Early stopping at epoch " << epoch << std::endl;
        break;
    }
}
```
</details>

<details>
<summary><b>Batch Processing</b> (click to expand)</summary>

```cpp
#include "nanoml/validation.hpp"

// Create minibatch generator
ml::MinibatchGenerator batcher(X_train, y_train, 32); // batch_size of 32

// Training loop with minibatches
for (int epoch = 0; epoch < max_epochs; ++epoch) {
    batcher.reset();  // Shuffle data for new epoch
    
    while (batcher.has_next_batch()) {
        auto [X_batch, y_batch] = batcher.next_batch();
        
        // Train on this batch
        // ...
    }
}
```
</details>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/KristiDodaj">Kristi Dodaj</a></p> 
</div>