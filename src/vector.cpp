#include "nanoml/vector.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ml {
    Vector::Vector(std::size_t n, double val) : data_(n, val) {}

    Vector::Vector(std::initializer_list<double> init) : data_(init) {}

    Vector::Vector(const std::vector<double>& data) : data_(data) {}

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

    bool Vector::operator==(const Vector& rhs) const {
        if (size() != rhs.size()) return false;
        
        for (std::size_t i = 0; i < size(); ++i) {
            if (data_[i] != rhs[i]) return false;
        }
        return true;
    }

    bool Vector::operator!=(const Vector& rhs) const {
        return !(*this == rhs);
    }

    double Vector::length() const {
        double sum_squares = 0.0;
        for (double val : data_) {
            sum_squares += val * val;
        }
        return std::sqrt(sum_squares);
    }

    Vector& Vector::normalize() {
        double len = length();
        assert(len > 0);

        return *this /= len;
    }

    Vector Vector::normalized() const {
        Vector result = *this;
        result.normalize();
        return result;
    }

    Vector operator*(double scalar, const Vector& vec) {
        return vec * scalar;
    }

    std::ostream& operator<<(std::ostream& os, const Vector& vec) {
        os << "[";
        const auto& data = vec.data();
        for (std::size_t i = 0; i < data.size(); ++i) {
            os << data[i];
            if (i < data.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }

    std::istream& operator>>(std::istream& is, Vector& vec) {
        std::vector<double> values;
        double val;
        char c;

        is >> c;
        if (c != '[') {
            is.setstate(std::ios::failbit);
            return is;
        }

        while (is.peek() != ']' && is.good()) {
            if (is >> val) {
                values.push_back(val);
 
                is >> c;
                if (c == ']') break;
                if (c != ',') {
                    is.setstate(std::ios::failbit);
                    break;
                }
            } else {
                is.setstate(std::ios::failbit);
                break;
            }
        }

        if (is.good()) {
            vec = Vector(values);
        }
        
        return is;
    }
}