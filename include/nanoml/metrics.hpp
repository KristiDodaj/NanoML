#pragma once
#include "nanoml/vector.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <map>

namespace ml {

    // ===============================================
    // Regression Metrics
    // ===============================================

    /**
     * Calculate Mean Squared Error (MSE)
     * @param y_hat Predicted values
     * @param y True values
     * @return MSE score
     */
    double mean_squared_error(const Vector& y_hat, const Vector& y);

    /**
     * Calculate R-squared (Coefficient of Determination)
     * @param y_hat Predicted values
     * @param y True values
     * @return R-squared score
     */
    double r_squared(const Vector& y_hat, const Vector& y);

    // ===============================================
    // Classification Metrics
    // ===============================================

    /**
     * Calculate accuracy for classification tasks.
     * Assumes y_hat contains probabilities or logits, applies threshold 0.5.
     * @param y_hat Predicted values (probabilities or logits)
     * @param y True labels (0 or 1)
     * @return Accuracy score (0.0 to 1.0)
     */
    double accuracy(const Vector& y_hat, const Vector& y);

    /**
     * Calculate precision for binary classification.
     * Assumes y_hat contains probabilities or logits, applies threshold 0.5.
     * @param y_hat Predicted values
     * @param y True labels (0 or 1)
     * @return Precision score
     */
    double precision(const Vector& y_hat, const Vector& y);

    /**
     * Calculate recall (sensitivity) for binary classification.
     * Assumes y_hat contains probabilities or logits, applies threshold 0.5.
     * @param y_hat Predicted values
     * @param y True labels (0 or 1)
     * @return Recall score
     */
    double recall(const Vector& y_hat, const Vector& y);

    /**
     * Calculate F1-score for binary classification.
     * Assumes y_hat contains probabilities or logits, applies threshold 0.5.
     * @param y_hat Predicted values
     * @param y True labels (0 or 1)
     * @return F1 score
     */
    double f1_score(const Vector& y_hat, const Vector& y);

    /**
     * Compute the confusion matrix for binary classification.
     * Assumes y_hat contains probabilities or logits, applies threshold 0.5.
     * @param y_hat Predicted values
     * @param y True labels (0 or 1)
     * @return A map representing the confusion matrix: { "tp": count, "fp": count, "tn": count, "fn": count }
     */
    std::map<std::string, int> confusion_matrix(const Vector& y_hat, const Vector& y);

}
