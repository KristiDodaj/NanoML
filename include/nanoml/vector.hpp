#pragma once
#include <vector>
#include <cstddef>

namespace ml {

class Vector {
    public:
        explicit Vector(std::size_t n, double val = 0.0);

        double  operator[](std::size_t i) const;
        double& operator[](std::size_t i);

        [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

        Vector& operator+=(const Vector& rhs);
        Vector  operator+(const Vector& rhs) const;

        Vector& operator-=(const Vector& rhs);
        Vector  operator-(const Vector& rhs) const;

        Vector& operator*=(double scalar);
        Vector  operator*(double scalar) const;

        Vector& operator/=(double scalar);
        Vector  operator/(double scalar) const;

        [[nodiscard]] double sum() const;
        [[nodiscard]] double dot(const Vector& rhs) const;

    private:
        std::vector<double> data_;
    };
}
