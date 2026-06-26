#include "numpp/numpp.hpp"
#include "numpp_test.hpp"

#include <cmath>

using namespace numpp;
using namespace numpp::random;

namespace {
constexpr int64_t N = 200000;

double smean(const ndarray& a) {
  const double* p = a.typed_data<double>();
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += p[i];
  return s / static_cast<double>(a.size());
}
double sstd(const ndarray& a) {
  const double m = smean(a);
  const double* p = a.typed_data<double>();
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += (p[i] - m) * (p[i] - m);
  return std::sqrt(s / static_cast<double>(a.size()));
}
double frac_within(const ndarray& a, double lo, double hi) {
  const double* p = a.typed_data<double>();
  int64_t c = 0;
  for (int64_t i = 0; i < a.size(); ++i)
    if (p[i] >= lo && p[i] <= hi) ++c;
  return static_cast<double>(c) / static_cast<double>(a.size());
}
bool near(double got, double exp, double tol) { return std::abs(got - exp) <= tol; }
}  // namespace

TEST_CASE("laplace moments") {
  Generator g(42);
  ndarray x = laplace(g, 0.0, 2.0, {N});
  CHECK(near(smean(x), 0.0, 0.05));
  CHECK(near(sstd(x), 2.0 * std::sqrt(2.0), 0.08));
}
TEST_CASE("logistic moments") {
  Generator g(42);
  ndarray x = logistic(g, 1.0, 0.5, {N});
  CHECK(near(smean(x), 1.0, 0.02));
  CHECK(near(sstd(x), 0.5 * M_PI / std::sqrt(3.0), 0.03));
}
TEST_CASE("gumbel moments") {
  Generator g(42);
  ndarray x = gumbel(g, 0.0, 1.0, {N});
  CHECK(near(smean(x), 0.5772156649, 0.03));
  CHECK(near(sstd(x), M_PI / std::sqrt(6.0), 0.04));
}
TEST_CASE("rayleigh moments") {
  Generator g(42);
  ndarray x = rayleigh(g, 2.0, {N});
  CHECK(near(smean(x), 2.0 * std::sqrt(M_PI / 2.0), 0.03));
  CHECK(near(sstd(x), 2.0 * std::sqrt((4.0 - M_PI) / 2.0), 0.03));
}
TEST_CASE("weibull moments") {
  Generator g(42);
  ndarray x = weibull(g, 1.5, {N});
  const double m = std::tgamma(1.0 + 1.0 / 1.5);
  const double v = std::tgamma(1.0 + 2.0 / 1.5) - m * m;
  CHECK(near(smean(x), m, 0.02));
  CHECK(near(sstd(x), std::sqrt(v), 0.02));
}
TEST_CASE("pareto moments") {
  Generator g(42);
  ndarray x = pareto(g, 3.0, {N});  // Lomax: mean 1/(a-1), var a/((a-1)^2 (a-2))
  CHECK(near(smean(x), 0.5, 0.03));
  CHECK(near(sstd(x), std::sqrt(0.75), 0.10));
}
TEST_CASE("power moments") {
  Generator g(42);
  ndarray x = power(g, 2.0, {N});
  CHECK(near(smean(x), 2.0 / 3.0, 0.01));
  CHECK(near(sstd(x), std::sqrt(2.0 / 36.0), 0.01));
  CHECK(frac_within(x, 0.0, 1.0) > 0.999);  // support [0,1]
}
TEST_CASE("standard_cauchy median and tail") {
  Generator g(42);
  ndarray x = standard_cauchy(g, {N});
  CHECK(near(frac_within(x, -1.0, 1.0), 0.5, 0.01));  // P(|X|<1) = 0.5
}
TEST_CASE("triangular moments") {
  Generator g(42);
  ndarray x = triangular(g, 0.0, 1.0, 3.0, {N});
  CHECK(near(smean(x), 4.0 / 3.0, 0.02));
  CHECK(near(sstd(x), std::sqrt(7.0 / 18.0), 0.02));
  CHECK(frac_within(x, 0.0, 3.0) > 0.999);  // support [left, right]
}
TEST_CASE("lognormal moments") {
  Generator g(42);
  ndarray x = lognormal(g, 0.0, 0.5, {N});
  const double mean = std::exp(0.125);
  const double var = (std::exp(0.25) - 1.0) * std::exp(0.25);
  CHECK(near(smean(x), mean, 0.02));
  CHECK(near(sstd(x), std::sqrt(var), 0.03));
}
