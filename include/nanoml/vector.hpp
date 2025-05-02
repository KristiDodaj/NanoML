#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <cmath>

namespace ml {
    class Vector {
        public:
            Vector() = default;
            explicit Vector(std::size_t n, double val = 0.0);
            Vector(std::initializer_list<double> init);
            Vector(const std::vector<double>& data);

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

            bool operator==(const Vector& rhs) const;
            bool operator!=(const Vector& rhs) const;

            [[nodiscard]] double sum() const;
            [[nodiscard]] double dot(const Vector& rhs) const;
            [[nodiscard]] double length() const;
            Vector& normalize();
            [[nodiscard]] Vector normalized() const;

            [[nodiscard]] const std::vector<double>& data() const { return data_; }

        private:
            std::vector<double> data_;
        };

        Vector operator*(double scalar, const Vector& vec);

        std::ostream& operator<<(std::ostream& os, const Vector& vec);
        std::istream& operator>>(std::istream& is, Vector& vec);
}
