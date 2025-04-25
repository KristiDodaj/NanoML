#include "nanoml/vector.hpp"
#include <cassert>  

namespace ml {
    Vector::Vector(std::size_t n, double val) : data_(n, val) {}

    double Vector::operator[](std::size_t i) const {
        assert(i < data_.size());
        return data_[i];
    }

    double& Vector::operator[](std::size_t i) {
        assert(i < data_.size());
        return data_[i];
    }

    Vector& Vector::operator+=(const Vector& rhs) {
        assert(size() == rhs.size());
        for (std::size_t i = 0; i < size(); ++i)
            data_[i] += rhs[i];
        return *this;
    }

    Vector Vector::operator+(const Vector& rhs) const { 
        assert(size() == rhs.size());
        Vector result = *this;
        result += rhs;
        return result;
    }

    Vector& Vector::operator-=(const Vector& rhs) {
        assert(size() == rhs.size());
        for (std::size_t i = 0; i < size(); ++i)
            data_[i] -= rhs[i];
        return *this;
    }

    Vector Vector::operator-(const Vector& rhs) const {
        assert(size() == rhs.size());
        Vector result = *this;
        result -= rhs;
        return result;
    }

    Vector& Vector::operator*=(double scalar) {
        for (auto& val : data_) val *= scalar;
        return *this;
    }

    Vector Vector::operator*(double scalar) const {
        Vector result = *this;
        result *= scalar;
        return result;
    }

    Vector& Vector::operator/=(double scalar) {
        assert(scalar != 0);
        for (auto& val : data_) val /= scalar;
        return *this;
    }

    Vector Vector::operator/(double scalar) const {
        assert(scalar != 0);
        Vector result = *this;
        result /= scalar;
        return result;
    }

    double Vector::sum() const {
        double total = 0.0;
        for (auto x : data_) total += x;
        return total;
    }

    double Vector::dot(const Vector& rhs) const {
        assert(size() == rhs.size());
        double total = 0.0;
        for (std::size_t i = 0; i < size(); ++i)
            total += data_[i] * rhs[i];
        return total;
    }
}