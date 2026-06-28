// Downstream-consumption smoke test. Includes the umbrella header on purpose so
// the build fails if any header it transitively pulls (e.g. numpp/interop/dlpack.h)
// is missing from an installed prefix (regression guard for #107). Used by both the
// find_package and add_subdirectory consumer projects under tests/packaging/.
#include <numpp/numpp.hpp>

#include <cstdio>

int main() {
  numpp::ndarray a = numpp::arange(0.0, 6.0, 1.0, numpp::kFloat64).reshape({2, 3});
  numpp::ndarray b = numpp::add(a, a);
  // 0+0 .. 5+5 summed = 2*(0+1+2+3+4+5) = 30
  const double total = numpp::sum(b).item<double>({});
  std::printf("numpp %s consumed OK, sum=%g\n", numpp::version(), total);
  return (total == 30.0) ? 0 : 1;
}
