#pragma once
#include "nanoml/validation.hpp"
#include <numeric>
#include <algorithm>
#include <iterator>
#include <cassert>

namespace ml {
    inline std::vector<size_t> random_indices(size_t size) {
        std::vector<size_t> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 gen(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), gen);
        return indices;
    }

    inline void train_test_split(
        const Matrix& X, const Vector& y,
        Matrix& X_train, Vector& y_train,
        Matrix& X_val, Vector& y_val,
        double test_size) {
        size_t n = X.rows();
        assert(n == y.size());
        size_t val_n = static_cast<size_t>(n * test_size);
        size_t train_n = n - val_n;
        auto indices = random_indices(n);
        X_train = Matrix(train_n, X.cols());
        y_train = Vector(train_n);
        X_val = Matrix(val_n, X.cols());
        y_val = Vector(val_n);
        for (size_t i = 0; i < train_n; ++i) {
            size_t idx = indices[i];
            for (size_t j = 0; j < X.cols(); ++j) X_train(i,j) = X(idx,j);
            y_train[i] = y[idx];
        }
        for (size_t i = 0; i < val_n; ++i) {
            size_t idx = indices[train_n + i];
            for (size_t j = 0; j < X.cols(); ++j) X_val(i,j) = X(idx,j);
            y_val[i] = y[idx];
        }
    }

    inline MinibatchGenerator::MinibatchGenerator(const Matrix& X, const Vector& y, size_t batch_size)
        : X_(X), y_(y), batch_size_(batch_size), indices_(random_indices(X.rows())), current_idx_(0) {}

    inline void MinibatchGenerator::reset() {
        indices_ = random_indices(X_.rows());
        current_idx_ = 0;
    }

    inline bool MinibatchGenerator::has_next_batch() const {
        return current_idx_ < indices_.size();
    }

    inline std::pair<Matrix, Vector> MinibatchGenerator::next_batch() {
        size_t start = current_idx_;
        size_t end = std::min(start + batch_size_, indices_.size());
        size_t bs = end - start;
        Matrix Xb(bs, X_.cols());
        Vector yb(bs);
        for (size_t i = 0; i < bs; ++i) {
            size_t idx = indices_[start + i];
            for (size_t j = 0; j < X_.cols(); ++j) Xb(i,j) = X_(idx,j);
            yb[i] = y_[idx];
        }
        current_idx_ = end;
        return {Xb, yb};
    }

    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    std::vector<double> cross_validate_regression(
        const Matrix& X, const Vector& y,
        int n_folds, double learning_rate, size_t epochs) {
        size_t n = X.rows();
        assert(n == y.size() && n_folds > 1);
        auto indices = random_indices(n);
        std::vector<double> scores;
        size_t fold_size = n / n_folds;
        for (int f = 0; f < n_folds; ++f) {
            Matrix X_train, X_val;
            Vector y_train, y_val;
            size_t start = f * fold_size;
            size_t end = (f == n_folds - 1) ? n : start + fold_size;
            size_t train_n = n - (end - start);
            X_train = Matrix(train_n, X.cols());
            y_train = Vector(train_n);
            X_val = Matrix(end - start, X.cols());
            y_val = Vector(end - start);
            size_t ti = 0, vi = 0;
            for (size_t i = 0; i < n; ++i) {
                size_t idx = indices[i];
                if (i >= start && i < end) {
                    for (size_t j = 0; j < X.cols(); ++j) X_val(vi,j) = X(idx,j);
                    y_val[vi++] = y[idx];
                } else {
                    for (size_t j = 0; j < X.cols(); ++j) X_train(ti,j) = X(idx,j);
                    y_train[ti++] = y[idx];
                }
            }
            ModelType model(X_train.cols());
            for (size_t e = 0; e < epochs; ++e) {
                auto y_hat = model.forward(X_train);
                auto grad = LossGradFunc()(y_hat, y_train);
                model.backward(X_train, grad, learning_rate);
            }
            auto y_pred = model.forward(X_val);
            scores.push_back(r_squared(y_pred, y_val));
        }
        return scores;
    }

    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    std::vector<double> cross_validate_classification(
        const Matrix& X, const Vector& y,
        int n_folds, double learning_rate, size_t epochs) {
        size_t n = X.rows();
        assert(n == y.size() && n_folds > 1);
        auto indices = random_indices(n);
        std::vector<double> scores;
        size_t fold_size = n / n_folds;
        for (int f = 0; f < n_folds; ++f) {
            Matrix X_train, X_val;
            Vector y_train, y_val;
            size_t start = f * fold_size;
            size_t end = (f == n_folds - 1) ? n : start + fold_size;
            size_t train_n = n - (end - start);
            X_train = Matrix(train_n, X.cols());
            y_train = Vector(train_n);
            X_val = Matrix(end - start, X.cols());
            y_val = Vector(end - start);
            size_t ti = 0, vi = 0;
            for (size_t i = 0; i < n; ++i) {
                size_t idx = indices[i];
                if (i >= start && i < end) {
                    for (size_t j = 0; j < X.cols(); ++j) X_val(vi,j) = X(idx,j);
                    y_val[vi++] = y[idx];
                } else {
                    for (size_t j = 0; j < X.cols(); ++j) X_train(ti,j) = X(idx,j);
                    y_train[ti++] = y[idx];
                }
            }
            ModelType model(X_train.cols());
            for (size_t e = 0; e < epochs; ++e) {
                auto y_hat = model.forward(X_train);
                auto grad = LossGradFunc()(y_hat, y_train);
                model.backward(X_train, grad, learning_rate);
            }
            auto y_pred = model.forward(X_val);
            scores.push_back(accuracy(y_pred, y_val));
        }
        return scores;
    }

    template <typename ModelType, typename LossFunc, typename LossGradFunc>
    void train_with_minibatches(
        ModelType& model,
        const Matrix& X, const Vector& y,
        double learning_rate, size_t epochs, size_t batch_size) {
        for (size_t e = 0; e < epochs; ++e) {
            MinibatchGenerator gen(X, y, batch_size);
            gen.reset();
            while (gen.has_next_batch()) {
                auto batch = gen.next_batch();
                auto y_hat = model.forward(batch.first);
                auto grad = LossGradFunc()(y_hat, batch.second);
                model.backward(batch.first, grad, learning_rate);
            }
        }
    }
}