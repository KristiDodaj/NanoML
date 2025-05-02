#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <sstream>

// Helper function for approximate comparison of doubles
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

// Helper function for approximate comparison of vectors
bool vectors_approx_equal(const ml::Vector& a, const ml::Vector& b, double epsilon = 1e-9) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], epsilon)) return false;
    }
    return true;
}

int main() {
    std::cout << "=== Testing basic Vector functionality ===" << std::endl;
    
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

    // == NEW TESTS FOR ADDED FUNCTIONALITY ==
    std::cout << "=== Testing new Vector functionality ===" << std::endl;
    
    // Test initializer list constructor
    ml::Vector k = {3.0, 4.0, 5.0};
    assert(k.size() == 3);
    assert(approx_equal(k[0], 3.0));
    assert(approx_equal(k[1], 4.0));
    assert(approx_equal(k[2], 5.0));
    std::cout << "Initializer list constructor: OK" << std::endl;
    
    // Test std::vector constructor
    std::vector<double> vec_data = {6.0, 7.0, 8.0};
    ml::Vector l(vec_data);
    assert(l.size() == 3);
    assert(approx_equal(l[0], 6.0));
    assert(approx_equal(l[1], 7.0));
    assert(approx_equal(l[2], 8.0));
    std::cout << "std::vector constructor: OK" << std::endl;
    
    // Test comparison operators
    ml::Vector m(3, 2.0);
    assert(m == b); // b = {2, 2, 2}
    assert(!(m != b));
    assert(m != a); // a = {1, 5, 1}
    assert(!(m == a));
    std::cout << "Comparison operators: OK" << std::endl;
    
    // Test scalar * vector operator
    ml::Vector n = 3.0 * a; // a = {1, 5, 1}
    assert(n == g); // g = {3, 15, 3}
    std::cout << "Scalar * vector operator: OK" << std::endl;
    
    // Test length method
    ml::Vector o = {3.0, 4.0}; // 3-4-5 right triangle
    assert(approx_equal(o.length(), 5.0));
    std::cout << "Length method: OK" << std::endl;
    
    // Test normalize methods
    ml::Vector p = {3.0, 4.0};
    p.normalize();
    assert(approx_equal(p.length(), 1.0));
    assert(approx_equal(p[0], 3.0/5.0));
    assert(approx_equal(p[1], 4.0/5.0));
    
    ml::Vector q = {6.0, 8.0};
    ml::Vector q_norm = q.normalized();
    assert(approx_equal(q_norm.length(), 1.0));
    assert(approx_equal(q_norm[0], 6.0/10.0));
    assert(approx_equal(q_norm[1], 8.0/10.0));
    // q should remain unchanged
    assert(approx_equal(q[0], 6.0));
    assert(approx_equal(q[1], 8.0));
    std::cout << "Normalize methods: OK" << std::endl;
    
    // Test stream operators
    ml::Vector r = {1.5, 2.5, 3.5};
    std::stringstream ss;
    ss << r;
    std::string str = ss.str();
    assert(str == "[1.5, 2.5, 3.5]");
    std::cout << "Output stream operator: OK" << std::endl;
    
    // Test input stream operator
    std::stringstream ss2("[4.5, 5.5, 6.5]");
    ml::Vector s;
    ss2 >> s;
    assert(s.size() == 3);
    assert(approx_equal(s[0], 4.5));
    assert(approx_equal(s[1], 5.5));
    assert(approx_equal(s[2], 6.5));
    std::cout << "Input stream operator: OK" << std::endl;
    
    // Test edge cases
    
    // Empty vector
    ml::Vector empty;
    assert(empty.size() == 0);
    assert(approx_equal(empty.sum(), 0.0));
    std::cout << "Empty vector: OK" << std::endl;
    
    // Single element vector
    ml::Vector single = {7.0};
    assert(single.size() == 1);
    assert(approx_equal(single.length(), 7.0));
    single.normalize();
    assert(approx_equal(single[0], 1.0));
    std::cout << "Single element vector: OK" << std::endl;

    std::cout << "\nAll Vector tests passed ✅\n";
    return 0;
}
