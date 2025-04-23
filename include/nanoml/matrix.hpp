#pragma once
#include <vector>
#include <cstddef>
#include "nanoml/vector.hpp"

namespace ml {

    class Matrix {
        public:
            Matrix(std::size_t rows, std::size_t cols, double val = 0.0);

            double operator()(std::size_t r, std::size_t c) const;
            double& operator()(std::size_t r, std::size_t c);

            [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
            [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
            [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

            Matrix& operator+=(const Matrix& rhs);
            Matrix operator+(const Matrix& rhs) const;
            Matrix& operator-=(const Matrix& rhs);
            Matrix operator-(const Matrix& rhs) const;

            Matrix& operator*=(double scalar);
            Matrix operator*(double scalar) const;
            Matrix& operator/=(double scalar);
            Matrix operator/(double scalar) const;

            Vector operator*(const Vector& v) const;
            Matrix operator*(const Matrix& rhs) const;
            Matrix transpose() const;
        
        private:
            std::size_t rows_, cols_;
            std::vector<double> data_;
            inline std::size_t idx(std::size_t r, std::size_t c) const noexcept{
                return r * cols_ + c;
            }
    };
}