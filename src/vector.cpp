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

}