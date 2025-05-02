#include "nanoml/linear_regression.hpp"
#include <cassert>
#include <fstream>
#include <string>

namespace  ml {
    LinearRegression::LinearRegression(std::size_t n_features)
        : W(n_features, 0.0), b(0.0) {}

    Vector LinearRegression::forward(const Matrix& X) const {
        assert(X.cols() == W.size());
        Vector y_hat(X.rows(), 0.0);
        for (std::size_t i = 0; i < X.rows(); ++i) {
            double dot = 0.0;
            for (std::size_t k = 0; k < X.cols(); ++k)
                dot += X(i,k) * W[k];
            y_hat[i] = dot + b;
        }
        return y_hat;
    }

    void LinearRegression::backward(const Matrix& X,
                                     const Vector& dLdy,
                                     double lr) {
        assert(X.rows() == dLdy.size());
        assert(X.cols() == W.size());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t k = 0; k < X.cols(); ++k)
                W[k] -= lr * dLdy[i] * X(i, k);
        }

        for (std::size_t i = 0; i < X.rows(); ++i)
            b -= lr * dLdy[i];
    }
    
    bool LinearRegression::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        std::string model_type = get_model_type();
        std::size_t type_size = model_type.size();
        file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
        file.write(model_type.c_str(), type_size);

        std::size_t n_features = W.size();
        file.write(reinterpret_cast<const char*>(&n_features), sizeof(n_features));

        file.write(reinterpret_cast<const char*>(&b), sizeof(b));

        const auto& data = W.data();
        file.write(reinterpret_cast<const char*>(data.data()), n_features * sizeof(double));
        
        return file.good();
    }
    
    bool LinearRegression::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        std::size_t type_size;
        file.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
        
        std::string model_type(type_size, '\0');
        file.read(&model_type[0], type_size);
        
        if (model_type != get_model_type()) {
            return false;
        }

        std::size_t n_features;
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));

        file.read(reinterpret_cast<char*>(&b), sizeof(b));

        W = Vector(n_features, 0.0);
        std::vector<double> temp_data(n_features);
        file.read(reinterpret_cast<char*>(temp_data.data()), n_features * sizeof(double));

        for (std::size_t i = 0; i < n_features; ++i) {
            W[i] = temp_data[i];
        }
        
        return file.good();
    }
}