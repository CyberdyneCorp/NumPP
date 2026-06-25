// Single-include usage of NumPP.
#include <cstdio>

#include "numpp/numpp.hpp"

int main() {
  using namespace numpp;
  std::printf("NumPP %s\n", version());

  ndarray a = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3});  // [[0,1,2],[3,4,5]]
  ndarray b = a.transpose();                                    // 3x2 view, no copy
  ndarray c = matmul(a, b);                                     // 2x2, CPU or BLAS
  std::printf("c[0,0]=%.1f c[1,1]=%.1f served-by=%s\n",
              c.item<double>({0, 0}), c.item<double>({1, 1}), backend_name(last_backend()));
  return 0;
}
