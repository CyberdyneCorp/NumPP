#include "numpp/random/distributions2.hpp"

#include "numpp/linalg/linalg.hpp"
#include "numpp/backend/backend.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <functional>
#include <vector>


namespace numpp {

namespace random {
namespace {

// Fill an int64 ndarray of the given size by repeatedly calling `draw`.
template <class F>
ndarray fill_int(const Shape& size, F draw) {
  ndarray out(size, kInt64, Order::C);
  int64_t* p = out.size() ? out.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) p[i] = draw();
  return out;
}

// Fill a float64 ndarray of the given size by repeatedly calling `draw`.
template <class F>
ndarray fill_double(const Shape& size, F draw) {
  ndarray out(size, kFloat64, Order::C);
  double* p = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) p[i] = draw();
  return out;
}

// Poisson(lam) via Knuth's multiplication algorithm.
int64_t poisson_knuth(Generator& g, double lam) {
  if (lam <= 0.0) return 0;
  const double L = std::exp(-lam);
  int64_t k = 0;
  double prod = 1.0;
  do {
    ++k;
    prod *= g.next_double();
  } while (prod > L);
  return k - 1;
}

double geometric_scalar(Generator& g, double p) {
  const double u = g.next_double();
  return std::floor(std::log(u) / std::log1p(-p)) + 1.0;
}

int64_t negbinom_scalar(Generator& g, double n, double p) {
  const double lam = g.next_std_gamma(n) * (1.0 - p) / p;
  return poisson_knuth(g, lam);
}

double zipf_scalar(Generator& g, double a) {
  const double am1 = a - 1.0;
  const double b = std::pow(2.0, am1);
  for (;;) {
    const double u = g.next_double();
    const double v = g.next_double();
    const double x = std::floor(std::pow(u, -1.0 / am1));
    if (x < 1.0) continue;
    const double t = std::pow(1.0 + 1.0 / x, am1);
    if (v * x * (t - 1.0) / (b - 1.0) <= t / b) return x;
  }
}

double logseries_scalar(Generator& g, double p) {
  const double r = std::log1p(-p);
  for (;;) {
    const double v = g.next_double();
    if (v >= p) return 1.0;
    const double u = g.next_double();
    const double q = -std::expm1(r * u);
    if (v <= q * q) return std::floor(1.0 + std::log(v) / std::log(q));
    if (v <= q) return 1.0;
    return 2.0;
  }
}

double standard_t_scalar(Generator& g, double df) {
  const double z = g.next_gauss();
  const double chi2 = 2.0 * g.next_std_gamma(df / 2.0);
  return z / std::sqrt(chi2 / df);
}

double f_scalar(Generator& g, double d1, double d2) {
  const double num = 2.0 * g.next_std_gamma(d1 / 2.0) / d1;
  const double den = 2.0 * g.next_std_gamma(d2 / 2.0) / d2;
  return num / den;
}

double wald_scalar(Generator& g, double mu, double lam) {
  double y = g.next_gauss();
  y *= y;
  const double x = mu + mu * mu * y / (2.0 * lam) -
                   mu / (2.0 * lam) * std::sqrt(4.0 * mu * lam * y + mu * mu * y * y);
  const double u = g.next_double();
  return u <= mu / (mu + x) ? x : mu * mu / x;
}

double vonmises_scalar(Generator& g, double mu, double kappa) {
  const double tau = 1.0 + std::sqrt(1.0 + 4.0 * kappa * kappa);
  const double rho = (tau - std::sqrt(2.0 * tau)) / (2.0 * kappa);
  const double r = (1.0 + rho * rho) / (2.0 * rho);
  for (;;) {
    const double u1 = g.next_double();
    const double z = std::cos(M_PI * u1);
    const double w = (1.0 + r * z) / (r + z);
    const double u2 = g.next_double();
    if (u2 < w * (2.0 - w) || u2 <= w * std::exp(1.0 - w)) {
      const double u3 = g.next_double();
      double res = (u3 < 0.5 ? -1.0 : 1.0) * std::acos(w) + mu;
      double wrapped = std::fmod(res + M_PI, 2.0 * M_PI);
      if (wrapped < 0.0) wrapped += 2.0 * M_PI;
      return wrapped - M_PI;
    }
  }
}

double noncentral_chisquare_scalar(Generator& g, double df, double nonc) {
  if (df > 1.0) {
    const double n = g.next_gauss() + std::sqrt(nonc);
    const double chi = 2.0 * g.next_std_gamma((df - 1.0) / 2.0);
    return n * n + chi;
  }
  const int64_t i = poisson_knuth(g, nonc / 2.0);
  return 2.0 * g.next_std_gamma(df / 2.0 + static_cast<double>(i));
}

}  // namespace

ndarray geometric(Generator& g, double p, const Shape& size) {
  return fill_int(size, [&]() { return static_cast<int64_t>(geometric_scalar(g, p)); });
}

ndarray negative_binomial(Generator& g, double n, double p, const Shape& size) {
  return fill_int(size, [&]() { return negbinom_scalar(g, n, p); });
}

ndarray zipf(Generator& g, double a, const Shape& size) {
  return fill_int(size, [&]() { return static_cast<int64_t>(zipf_scalar(g, a)); });
}

ndarray logseries(Generator& g, double p, const Shape& size) {
  return fill_int(size, [&]() { return static_cast<int64_t>(logseries_scalar(g, p)); });
}

ndarray standard_t(Generator& g, double df, const Shape& size) {
  return fill_double(size, [&]() { return standard_t_scalar(g, df); });
}

ndarray f(Generator& g, double dfnum, double dfden, const Shape& size) {
  return fill_double(size, [&]() { return f_scalar(g, dfnum, dfden); });
}

ndarray wald(Generator& g, double mean, double scale, const Shape& size) {
  return fill_double(size, [&]() { return wald_scalar(g, mean, scale); });
}

ndarray vonmises(Generator& g, double mu, double kappa, const Shape& size) {
  return fill_double(size, [&]() { return vonmises_scalar(g, mu, kappa); });
}

ndarray noncentral_chisquare(Generator& g, double df, double nonc, const Shape& size) {
  return fill_double(size, [&]() { return noncentral_chisquare_scalar(g, df, nonc); });
}

ndarray multinomial(Generator& g, int64_t n, const std::vector<double>& pvals) {
  const int64_t k = static_cast<int64_t>(pvals.size());
  ndarray out(Shape{k}, kInt64, Order::C);
  int64_t* o = k ? out.typed_data<int64_t>() : nullptr;
  int64_t remaining = n;
  double remp = 1.0;
  for (int64_t i = 0; i < k; ++i) {
    if (i == k - 1) {
      o[i] = remaining;
      break;
    }
    double pi = remp > 0.0 ? pvals[i] / remp : 0.0;
    if (pi > 1.0) pi = 1.0;
    if (pi < 0.0) pi = 0.0;
    const int64_t ki = remaining > 0
        ? g.binomial(remaining, pi, Shape{1}).item<int64_t>({0})
        : 0;
    o[i] = ki;
    remaining -= ki;
    remp -= pvals[i];
  }
  return out;
}

ndarray multivariate_normal(Generator& g, const ndarray& mean, const ndarray& cov, int64_t size) {
  const int64_t d = mean.size();
  ndarray L = linalg::cholesky(cov.astype(kFloat64));
  ndarray Z = g.standard_normal(Shape{size, d});
  ndarray X = matmul(Z, L.transpose()).ascontiguousarray();
  ndarray m = mean.astype(kFloat64).ascontiguousarray();
  const double* mp = d ? m.typed_data<double>() : nullptr;
  double* xp = X.size() ? X.typed_data<double>() : nullptr;
  for (int64_t r = 0; r < size; ++r)
    for (int64_t c = 0; c < d; ++c) xp[r * d + c] += mp[c];
  return X;
}

ndarray dirichlet(Generator& g, const std::vector<double>& alpha, int64_t size) {
  const int64_t k = static_cast<int64_t>(alpha.size());
  ndarray out(Shape{size, k}, kFloat64, Order::C);
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t r = 0; r < size; ++r) {
    double s = 0.0;
    for (int64_t j = 0; j < k; ++j) {
      const double y = g.next_std_gamma(alpha[j]);
      o[r * k + j] = y;
      s += y;
    }
    if (s > 0.0)
      for (int64_t j = 0; j < k; ++j) o[r * k + j] /= s;
  }
  return out;
}

}  // namespace random

}  // namespace numpp
