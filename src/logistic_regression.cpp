#include "nanoml/logistic_regression.hpp"
#include "nanoml/activations.hpp"
#include <cassert>
#include <fstream>
#include <string>

namespace ml {
    LogisticRegression::LogisticRegression(std::size_t n)
    : W(n, 0.0), b(0.0) {}

    Vector LogisticRegression::forward(const Matrix& X) const {
        assert(X.cols() == W.size());
        Vector out(X.rows());
        for (std::size_t i=0;i<X.rows();++i){
            double z = 0.0;
            for (std::size_t k=0;k<X.cols();++k) z += X(i,k)*W[k];
            out[i] = sigmoid(z + b);
        }
        return out;
    }

    void LogisticRegression::backward(const Matrix& X,
                                    const Vector& dLdy,
                                    double lr){
        Vector grad_w(W.size(),0.0);
        double grad_b = 0.0;
        for(std::size_t i=0;i<X.rows();++i){
            grad_b += dLdy[i];
            for(std::size_t k=0;k<X.cols();++k)
                grad_w[k] += X(i,k)*dLdy[i];
        }
        double m = static_cast<double>(X.rows());
        grad_w /= m;  grad_b /= m;

        W -= grad_w * lr;
        b -= grad_b * lr;
    }

    bool LogisticRegression::save(const std::string& filename) const {
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
    
    bool LogisticRegression::load(const std::string& filename) {
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
