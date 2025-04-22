#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>

int main() {
    ml::Vector a(3, 1.0), b(3, 2.0);
    ml::Vector c = a + b;            // should be {3,3,3}
    assert(c.size() == 3);
    assert(c[0] == 3.0 && c[2] == 3.0);
    std::cout << "Vector test passed ✅\n";
    return 0;
}
