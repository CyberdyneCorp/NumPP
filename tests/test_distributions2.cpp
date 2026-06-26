#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;

using numpp::random::Generator;

namespace {

double smean(const numpp::ndarray& a) {
  const double* p = a.typed_data<double>();
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += p[i];
  return s / static_cast<double>(a.size());
}
double smean_i(const numpp::ndarray& a) {
  const int64_t* p = a.typed_data<int64_t>();
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += static_cast<double>(p[i]);
  return s / static_cast<double>(a.size());
}
double sstd(const numpp::ndarray& a) {
  const double* p = a.typed_data<double>();
  double m = smean(a), s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) { double d = p[i] - m; s += d * d; }
  return std::sqrt(s / static_cast<double>(a.size()));
}
bool near(double got, double exp, double tol) { return std::abs(got - exp) <= tol; }

constexpr int64_t N = 200000;

}  // namespace

TEST_CASE("random.geometric mean ~ 1/p") {
  Generator g(42);
  auto a = numpp::random::geometric(g, 0.25, numpp::Shape{N});
  CHECK(near(smean_i(a), 1.0 / 0.25, 0.05));
}

TEST_CASE("random.negative_binomial mean ~ n(1-p)/p") {
  Generator g(42);
  double n = 10.0, p = 0.4;
  auto a = numpp::random::negative_binomial(g, n, p, numpp::Shape{N});
  CHECK(near(smean_i(a), n * (1.0 - p) / p, 0.1));
}

TEST_CASE("random.zipf all >= 1") {
  Generator g(42);
  auto a = numpp::random::zipf(g, 2.5, numpp::Shape{N});
  const int64_t* p = a.typed_data<int64_t>();
  bool ok = true;
  for (int64_t i = 0; i < a.size(); ++i) if (p[i] < 1) ok = false;
  CHECK(ok);
}

TEST_CASE("random.logseries all >= 1") {
  Generator g(42);
  auto a = numpp::random::logseries(g, 0.6, numpp::Shape{N});
  const int64_t* p = a.typed_data<int64_t>();
  bool ok = true;
  for (int64_t i = 0; i < a.size(); ++i) if (p[i] < 1) ok = false;
  CHECK(ok);
}

TEST_CASE("random.standard_t mean ~ 0, var ~ df/(df-2)") {
  Generator g(42);
  double df = 10.0;
  auto a = numpp::random::standard_t(g, df, numpp::Shape{N});
  CHECK(near(smean(a), 0.0, 0.05));
  double v = sstd(a) * sstd(a);
  CHECK(near(v, df / (df - 2.0), 0.1));
}

TEST_CASE("random.f mean ~ d2/(d2-2)") {
  Generator g(42);
  double d1 = 8.0, d2 = 20.0;
  auto a = numpp::random::f(g, d1, d2, numpp::Shape{N});
  CHECK(near(smean(a), d2 / (d2 - 2.0), 0.05));
}

TEST_CASE("random.wald mean ~ mu") {
  Generator g(42);
  double mu = 2.0, scale = 5.0;
  auto a = numpp::random::wald(g, mu, scale, numpp::Shape{N});
  CHECK(near(smean(a), mu, 0.05));
}

TEST_CASE("random.vonmises in (-pi,pi] and mean ~ mu for large kappa") {
  Generator g(42);
  double mu = 0.5, kappa = 50.0;
  auto a = numpp::random::vonmises(g, mu, kappa, numpp::Shape{N});
  const double* p = a.typed_data<double>();
  bool ok = true;
  for (int64_t i = 0; i < a.size(); ++i) if (p[i] <= -M_PI - 1e-12 || p[i] > M_PI + 1e-12) ok = false;
  CHECK(ok);
  CHECK(near(smean(a), mu, 0.02));
}

TEST_CASE("random.noncentral_chisquare mean ~ df+nonc") {
  Generator g(42);
  double df = 5.0, nonc = 3.0;
  auto a = numpp::random::noncentral_chisquare(g, df, nonc, numpp::Shape{N});
  CHECK(near(smean(a), df + nonc, 0.1));
}

TEST_CASE("random.multinomial counts sum to n") {
  Generator g(42);
  std::vector<double> pv{0.2, 0.5, 0.3};
  auto a = numpp::random::multinomial(g, 1000, pv);
  CHECK(a.size() == 3);
  const int64_t* p = a.typed_data<int64_t>();
  int64_t s = p[0] + p[1] + p[2];
  CHECK(s == 1000);
  // approximate proportions
  CHECK(near(static_cast<double>(p[1]) / 1000.0, 0.5, 0.05));
}

TEST_CASE("random.multivariate_normal column means ~ mean") {
  Generator g(42);
  numpp::ndarray mean(numpp::Shape{2}, numpp::kFloat64, numpp::Order::C);
  mean.set_item<double>({0}, 1.0);
  mean.set_item<double>({1}, -2.0);
  numpp::ndarray cov = numpp::eye(2, 2, 0, numpp::kFloat64);
  auto a = numpp::random::multivariate_normal(g, mean, cov, N);
  CHECK(a.shape()[0] == N);
  CHECK(a.shape()[1] == 2);
  const double* p = a.typed_data<double>();
  double m0 = 0.0, m1 = 0.0;
  for (int64_t i = 0; i < N; ++i) { m0 += p[i * 2]; m1 += p[i * 2 + 1]; }
  m0 /= N; m1 /= N;
  CHECK(near(m0, 1.0, 0.05));
  CHECK(near(m1, -2.0, 0.05));
}

TEST_CASE("random.dirichlet rows sum to 1, col means ~ alpha_j/sum") {
  Generator g(42);
  std::vector<double> al{1.0, 2.0, 3.0};
  double as = 6.0;
  auto a = numpp::random::dirichlet(g, al, N);
  CHECK(a.shape()[0] == N);
  CHECK(a.shape()[1] == 3);
  const double* p = a.typed_data<double>();
  bool sums_ok = true;
  double c0 = 0.0, c1 = 0.0, c2 = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    double s = p[i * 3] + p[i * 3 + 1] + p[i * 3 + 2];
    if (std::abs(s - 1.0) > 1e-9) sums_ok = false;
    c0 += p[i * 3]; c1 += p[i * 3 + 1]; c2 += p[i * 3 + 2];
  }
  CHECK(sums_ok);
  CHECK(near(c0 / N, al[0] / as, 0.02));
  CHECK(near(c1 / N, al[1] / as, 0.02));
  CHECK(near(c2 / N, al[2] / as, 0.02));
}
