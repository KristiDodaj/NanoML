#include "nanoml/matrix.hpp"
#include <cassert>  

namespace ml {

    Matrix::Matrix(std::size_t rows, std::size_t cols, double val) 
        : rows_(rows), cols_(cols), data_(rows * cols, val) {}
    
    double Matrix::operator()(std::size_t r, std::size_t c) const {
        assert(r < rows_ && c < cols_);
        return data_[idx(r, c)];
    }

    double& Matrix::operator()(std::size_t r, std::size_t c) {
        assert(r < rows_ && c < cols_);
        return data_[idx(r, c)];
    }

    Matrix& Matrix::operator+=(const Matrix& rhs) {
        assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            data_[i] += rhs.data_[i];
        return *this;
    }

    Matrix Matrix::operator+(const Matrix& rhs) const {
        assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
        Matrix result = *this;
        result += rhs;
        return result;
    }

    Matrix& Matrix::operator-=(const Matrix& rhs) {
        assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            data_[i] -= rhs.data_[i];
        return *this;
    }

    Matrix Matrix::operator-(const Matrix& rhs) const {
        assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
        Matrix result = *this;
        result -= rhs;
        return result;
    }

    Matrix& Matrix::operator*=(double scalar) {
        for (auto& val : data_) val *= scalar;
        return *this;
    }

    Matrix Matrix::operator*(double scalar) const {
        Matrix result = *this;
        result *= scalar;
        return result;
    }

    Matrix& Matrix::operator/=(double scalar) {
        assert(scalar != 0);
        for (auto& val : data_) val /= scalar;
        return *this;
    }

    Matrix Matrix::operator/(double scalar) const {
        assert(scalar != 0);
        Matrix result = *this;
        result /= scalar;
        return result;
    }

    Vector Matrix::operator*(const Vector& v) const {
        assert(cols_ == v.size());
        Vector result(rows_, 0.0);
        for (std::size_t r = 0; r < rows_; ++r) {
            result[r] = 0;
            for (std::size_t c = 0; c < cols_; ++c) {
                result[r] += (*this)(r, c) * v[c];
            }
        }
        return result;
    }

    Matrix Matrix::operator*(const Matrix& rhs) const {
        assert(cols_ == rhs.rows_);
        Matrix result(rows_, rhs.cols_, 0.0);
    
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < rhs.cols_; ++c)
                for (std::size_t k = 0; k < cols_; ++k)
                    result(r,c) += (*this)(r,k) * rhs(k,c);
    
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(cols_, rows_, 0.0);
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < cols_; ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }
}