#pragma once
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include "nanoml/model.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <numeric>

namespace ml {
    // ===============================================
    // Data Splitting Utilities
    // ===============================================
    
    /**
     * Generate a random permutation of indices
     * @param size Number of indices to generate
     * @return Vector of shuffled indices from 0 to size-1
     */
    std::vector<size_t> random_indices(size_t size);
    
    /**
     * Split data into training and validation/test sets
     * @param X Feature matrix
     * @param y Target vector
     * @param X_train Output parameter for training features
     * @param y_train Output parameter for training targets
     * @param X_val Output parameter for validation features
     * @param y_val Output parameter for validation targets
     * @param test_size Fraction of data to use for validation (default: 0.2)
     */
    void train_test_split(
        const Matrix& X, const Vector& y,
        Matrix& X_train, Vector& y_train,
        Matrix& X_val, Vector& y_val,
        double test_size = 0.2);
    
    // ===============================================
    // Minibatch Processing
    // ===============================================
    
    /**
     * @brief A class for generating minibatches from a dataset
     * 
     * This class provides functionality for creating minibatches from a dataset,
     * with support for shuffling data between epochs.
     */
    class MinibatchGenerator {
    private:
        const Matrix& X_;
        const Vector& y_;
        size_t batch_size_;
        std::vector<size_t> indices_;
        size_t current_idx_;
        
    public:
        /**
         * @brief Construct a new Minibatch Generator
         * 
         * @param X Feature matrix
         * @param y Target vector
         * @param batch_size Size of each batch
         */
        MinibatchGenerator(const Matrix& X, const Vector& y, size_t batch_size);
        
        /**
         * @brief Reset the generator for a new epoch
         * 
         * This shuffles the data for the next pass through the dataset
         */
        void reset();
        
        /**
         * @brief Check if there are more batches available
         * 
         * @return true if there are more batches, false otherwise
         */
        bool has_next_batch() const;
        
        /**
         * @brief Get the next minibatch
         * 
         * @return A pair containing the feature matrix and target vector for this batch
         */
        std::pair<Matrix, Vector> next_batch();
    };
    
    // ===============================================
    // Cross-Validation 
    // ===============================================
    
    /**
     * @brief K-fold cross-validation for regression models
     * 
     * @tparam ModelType Type of the model to train and evaluate
     * @tparam LossFunc Type of the loss function
     * @tparam LossGradFunc Type of the loss gradient function
     * @param X Feature matrix
     * @param y Target vector
     * @param n_folds Number of folds for cross-validation
     * @param learning_rate Learning rate for optimization
     * @param epochs Number of training epochs
     * @return Vector of R-squared scores for each fold
     */
    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    std::vector<double> cross_validate_regression(
        const Matrix& X, const Vector& y,
        int n_folds, double learning_rate, size_t epochs);
    
    /**
     * @brief K-fold cross-validation for classification models
     * 
     * @tparam ModelType Type of the model to train and evaluate
     * @tparam LossFunc Type of the loss function
     * @tparam LossGradFunc Type of the loss gradient function
     * @param X Feature matrix
     * @param y Target vector
     * @param n_folds Number of folds for cross-validation
     * @param learning_rate Learning rate for optimization
     * @param epochs Number of training epochs
     * @return Vector of accuracy scores for each fold
     */
    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    std::vector<double> cross_validate_classification(
        const Matrix& X, const Vector& y,
        int n_folds, double learning_rate, size_t epochs);
    
    // ===============================================
    // Minibatch Training 
    // ===============================================
    
    /**
     * @brief Train a model using minibatches
     * 
     * @tparam ModelType Type of the model to train
     * @tparam LossFunc Type of the loss function
     * @tparam LossGradFunc Type of the loss gradient function
     * @param model Model to train
     * @param X Feature matrix
     * @param y Target vector
     * @param learning_rate Learning rate for optimization
     * @param epochs Number of training epochs
     * @param batch_size Size of each minibatch
     */
    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    void train_with_minibatches(
        ModelType& model,
        const Matrix& X, const Vector& y,
        double learning_rate, size_t epochs, size_t batch_size);

}

#include "nanoml/validation.inl"