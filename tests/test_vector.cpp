#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>
#include <cmath> // For std::fabs

// Helper function for approximate comparison of doubles
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    // Test Constructor and operator[]
    ml::Vector a(3, 1.0);
    assert(a.size() == 3);
    assert(approx_equal(a[0], 1.0));
    assert(approx_equal(a[1], 1.0));
    assert(approx_equal(a[2], 1.0));

    // Test non-const operator[]
    a[1] = 5.0;
    assert(approx_equal(a[1], 5.0));

    ml::Vector b(3, 2.0);
    assert(b.size() == 3);
    assert(approx_equal(b[0], 2.0));
    assert(approx_equal(b[1], 2.0));
    assert(approx_equal(b[2], 2.0));

    // Test operator+
    ml::Vector c = a + b; // a = {1, 5, 1}, b = {2, 2, 2}
    assert(c.size() == 3);
    assert(approx_equal(c[0], 3.0)); // 1 + 2
    assert(approx_equal(c[1], 7.0)); // 5 + 2
    assert(approx_equal(c[2], 3.0)); // 1 + 2

    // Test operator+=
    ml::Vector d(3, 0.5);
    d += a; // d = {0.5, 0.5, 0.5} + {1, 5, 1}
    assert(approx_equal(d[0], 1.5));
    assert(approx_equal(d[1], 5.5));
    assert(approx_equal(d[2], 1.5));

    // Test operator-
    ml::Vector e = c - a; // c = {3, 7, 3}, a = {1, 5, 1}
    assert(e.size() == 3);
    assert(approx_equal(e[0], 2.0)); // 3 - 1
    assert(approx_equal(e[1], 2.0)); // 7 - 5
    assert(approx_equal(e[2], 2.0)); // 3 - 1
    // Check if e is now equal to b
    assert(approx_equal(e[0], b[0]));
    assert(approx_equal(e[1], b[1]));
    assert(approx_equal(e[2], b[2]));


    // Test operator-=
    ml::Vector f = c; // f = {3, 7, 3}
    f -= a; // f = {3, 7, 3} - {1, 5, 1}
    assert(approx_equal(f[0], 2.0));
    assert(approx_equal(f[1], 2.0));
    assert(approx_equal(f[2], 2.0));
    // Check if f is now equal to b
    assert(approx_equal(f[0], b[0]));
    assert(approx_equal(f[1], b[1]));
    assert(approx_equal(f[2], b[2]));

    // Test operator* (scalar)
    ml::Vector g = a * 3.0; // a = {1, 5, 1}
    assert(g.size() == 3);
    assert(approx_equal(g[0], 3.0)); // 1 * 3
    assert(approx_equal(g[1], 15.0)); // 5 * 3
    assert(approx_equal(g[2], 3.0)); // 1 * 3

    // Test operator*= (scalar)
    ml::Vector h = a; // h = {1, 5, 1}
    h *= 3.0;
    assert(approx_equal(h[0], 3.0));
    assert(approx_equal(h[1], 15.0));
    assert(approx_equal(h[2], 3.0));
    // Check if h is now equal to g
    assert(approx_equal(h[0], g[0]));
    assert(approx_equal(h[1], g[1]));
    assert(approx_equal(h[2], g[2]));

    // Test operator/ (scalar)
    ml::Vector i = b / 2.0; // b = {2, 2, 2}
    assert(i.size() == 3);
    assert(approx_equal(i[0], 1.0)); // 2 / 2
    assert(approx_equal(i[1], 1.0)); // 2 / 2
    assert(approx_equal(i[2], 1.0)); // 2 / 2

    // Test operator/= (scalar)
    ml::Vector j = b; // j = {2, 2, 2}
    j /= 2.0;
    assert(approx_equal(j[0], 1.0));
    assert(approx_equal(j[1], 1.0));
    assert(approx_equal(j[2], 1.0));
    // Check if j is now equal to i
    assert(approx_equal(j[0], i[0]));
    assert(approx_equal(j[1], i[1]));
    assert(approx_equal(j[2], i[2]));

    // Test sum()
    // a = {1, 5, 1}, b = {2, 2, 2}, c = {3, 7, 3}
    assert(approx_equal(a.sum(), 7.0)); // 1 + 5 + 1
    assert(approx_equal(b.sum(), 6.0)); // 2 + 2 + 2
    assert(approx_equal(c.sum(), 13.0)); // 3 + 7 + 3

    // Test dot()
    // a = {1, 5, 1}, b = {2, 2, 2}
    double dot_ab = a.dot(b); // (1*2) + (5*2) + (1*2) = 2 + 10 + 2 = 14
    assert(approx_equal(dot_ab, 14.0));

    double dot_bb = b.dot(b); // (2*2) + (2*2) + (2*2) = 4 + 4 + 4 = 12
    assert(approx_equal(dot_bb, 12.0));

    std::cout << "All Vector tests passed ✅\n";
    return 0;
}
