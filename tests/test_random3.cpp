#include <cmath>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"

using namespace numpp;
using numpp::random::Generator;

static void moments(const ndarray& a, double& m, double& var) {
  const int64_t n = a.size();
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += a.item<double>({i});
  m = s / n;
  double v = 0;
  for (int64_t i = 0; i < n; ++i) { double d = a.item<double>({i}) - m; v += d * d; }
  var = v / n;
}
static void moments_i(const ndarray& a, double& m, double& var) {
  const int64_t n = a.size();
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += a.item<int64_t>({i});
  m = s / n;
  double v = 0;
  for (int64_t i = 0; i < n; ++i) { double d = a.item<int64_t>({i}) - m; v += d * d; }
  var = v / n;
}

TEST_CASE("standard_normal moments (issue #8: statistical, not bit-exact)") {
  double m, v; moments(Generator(1).standard_normal(200000), m, v);
  CHECK(std::abs(m) < 0.02);
  CHECK(std::abs(v - 1.0) < 0.03);
}
TEST_CASE("standard_exponential moments") {
  double m, v; moments(Generator(2).standard_exponential({200000}), m, v);
  CHECK(std::abs(m - 1.0) < 0.03);     // mean 1, var 1
  CHECK(std::abs(v - 1.0) < 0.06);
}
TEST_CASE("normal(loc,scale) moments") {
  double m, v; moments(Generator(3).normal(5.0, 2.0, 200000), m, v);
  CHECK(std::abs(m - 5.0) < 0.05);
  CHECK(std::abs(v - 4.0) < 0.12);
}
TEST_CASE("exponential(scale) moments") {
  double m, v; moments(Generator(4).exponential(3.0, {200000}), m, v);
  CHECK(std::abs(m - 3.0) < 0.1);      // mean=scale, var=scale^2
  CHECK(std::abs(v - 9.0) < 0.6);
}
TEST_CASE("gamma moments (mean=k*theta, var=k*theta^2)") {
  double m, v; moments(Generator(5).gamma(2.0, 1.5, {200000}), m, v);
  CHECK(std::abs(m - 3.0) < 0.05);
  CHECK(std::abs(v - 4.5) < 0.2);
}
TEST_CASE("beta moments (mean=a/(a+b))") {
  double m, v; moments(Generator(6).beta(2.0, 5.0, {200000}), m, v);
  CHECK(std::abs(m - 2.0 / 7.0) < 0.01);
  CHECK(std::abs(v - (2.0 * 5.0) / (49.0 * 8.0)) < 0.005);  // ab/((a+b)^2 (a+b+1))
}
TEST_CASE("chisquare moments (mean=df, var=2df)") {
  double m, v; moments(Generator(7).chisquare(4.0, {200000}), m, v);
  CHECK(std::abs(m - 4.0) < 0.05);
  CHECK(std::abs(v - 8.0) < 0.4);
}
TEST_CASE("poisson moments (mean=var=lam)") {
  double m, v; moments_i(Generator(8).poisson(4.0, {200000}), m, v);
  CHECK(std::abs(m - 4.0) < 0.05);
  CHECK(std::abs(v - 4.0) < 0.1);
}
TEST_CASE("binomial moments (mean=np, var=np(1-p))") {
  double m, v; moments_i(Generator(9).binomial(20, 0.3, {200000}), m, v);
  CHECK(std::abs(m - 6.0) < 0.05);
  CHECK(std::abs(v - 4.2) < 0.1);
}
