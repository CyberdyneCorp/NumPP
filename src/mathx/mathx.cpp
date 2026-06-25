#include "numpp/mathx/mathx.hpp"

#include "numpp/core/creation.hpp"   // full
#include "numpp/core/shape.hpp"      // broadcast_shapes
#include "numpp/umath/ufunc.hpp"     // rint/rad2deg/deg2rad/multiply/divide/power

#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace numpp {
namespace {

ndarray unary_double(const ndarray& a, const std::function<double(double)>& f) {
  ndarray x = a.astype(kFloat64).ascontiguousarray();
  ndarray out(x.shape(), kFloat64, Order::C);
  const double* p = x.size() ? x.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < x.size(); ++i) o[i] = f(p[i]);
  return out;
}

// Broadcast a and b to a common float64 contiguous pair, returning the shared shape.
std::pair<ndarray, ndarray> broadcast_pair(const ndarray& a, const ndarray& b, Shape& sh) {
  sh = broadcast_shapes(a.shape(), b.shape());
  ndarray af = a.astype(kFloat64).broadcast_to(sh).copy();
  ndarray bf = b.astype(kFloat64).broadcast_to(sh).copy();
  return {af, bf};
}

ndarray binary_double(const ndarray& a, const ndarray& b, const std::function<double(double, double)>& f) {
  Shape sh;
  auto [af, bf] = broadcast_pair(a, b, sh);
  ndarray out(sh, kFloat64, Order::C);
  const double* pa = af.size() ? af.typed_data<double>() : nullptr;
  const double* pb = bf.size() ? bf.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) o[i] = f(pa[i], pb[i]);
  return out;
}

ndarray int_binary(const ndarray& a, const ndarray& b, const std::function<int64_t(int64_t, int64_t)>& f) {
  Shape sh = broadcast_shapes(a.shape(), b.shape());
  ndarray ai = a.astype(kInt64).broadcast_to(sh).copy();
  ndarray bi = b.astype(kInt64).broadcast_to(sh).copy();
  ndarray out(sh, kInt64, Order::C);
  const int64_t* pa = ai.size() ? ai.typed_data<int64_t>() : nullptr;
  const int64_t* pb = bi.size() ? bi.typed_data<int64_t>() : nullptr;
  int64_t* o = out.size() ? out.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) o[i] = f(pa[i], pb[i]);
  return out;
}

double i0_scalar(double x) {
  const double t = x * x / 4.0;
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 200; ++k) {
    term *= t / (static_cast<double>(k) * static_cast<double>(k));
    sum += term;
    if (term < 1e-18 * sum) break;
  }
  return sum;
}

}  // namespace

ndarray around(const ndarray& a, int64_t decimals) {
  if (decimals == 0) return rint(a.astype(kFloat64));
  const double scale = std::pow(10.0, static_cast<double>(decimals));
  ndarray s = full(Shape{}, scale, kFloat64);
  return divide(rint(multiply(a.astype(kFloat64), s)), s);
}

ndarray degrees(const ndarray& a) { return rad2deg(a); }
ndarray radians(const ndarray& a) { return deg2rad(a); }

ndarray sinc(const ndarray& a) {
  return unary_double(a, [](double x) {
    if (x == 0.0) return 1.0;
    const double px = M_PI * x;
    return std::sin(px) / px;
  });
}

ndarray gcd(const ndarray& a, const ndarray& b) {
  return int_binary(a, b, [](int64_t x, int64_t y) { return std::gcd(x, y); });
}
ndarray lcm(const ndarray& a, const ndarray& b) {
  return int_binary(a, b, [](int64_t x, int64_t y) { return std::lcm(x, y); });
}

ndarray nan_to_num(const ndarray& a, double nan, std::optional<double> posinf, std::optional<double> neginf) {
  const double big = std::numeric_limits<double>::max();
  const double pos = posinf.value_or(big);
  const double neg = neginf.value_or(-big);
  return unary_double(a, [=](double x) {
    if (std::isnan(x)) return nan;
    if (std::isinf(x)) return x > 0 ? pos : neg;
    return x;
  });
}

ndarray logaddexp(const ndarray& a, const ndarray& b) {
  return binary_double(a, b, [](double x, double y) {
    const double m = std::max(x, y);
    if (std::isinf(m)) return m;
    return m + std::log1p(std::exp(-std::abs(x - y)));
  });
}
ndarray logaddexp2(const ndarray& a, const ndarray& b) {
  const double inv = 1.0 / std::log(2.0);
  return binary_double(a, b, [inv](double x, double y) {
    const double m = std::max(x, y);
    if (std::isinf(m)) return m;
    return m + std::log1p(std::exp2(-std::abs(x - y))) * inv;
  });
}

ndarray float_power(const ndarray& a, const ndarray& b) {
  return power(a.astype(kFloat64), b.astype(kFloat64));
}
ndarray fmod(const ndarray& a, const ndarray& b) {
  return binary_double(a, b, [](double x, double y) { return std::fmod(x, y); });
}
ndarray heaviside(const ndarray& a, const ndarray& h0) {
  return binary_double(a, h0, [](double x, double h) {
    if (x < 0) return 0.0;
    if (x > 0) return 1.0;
    return h;
  });
}

std::pair<ndarray, ndarray> modf(const ndarray& a) {
  ndarray x = a.astype(kFloat64).ascontiguousarray();
  ndarray frac(x.shape(), kFloat64, Order::C), integ(x.shape(), kFloat64, Order::C);
  const double* p = x.size() ? x.typed_data<double>() : nullptr;
  double* pf = frac.size() ? frac.typed_data<double>() : nullptr;
  double* pi = integ.size() ? integ.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < x.size(); ++i) { double ip; pf[i] = std::modf(p[i], &ip); pi[i] = ip; }
  return {frac, integ};
}

std::pair<ndarray, ndarray> frexp(const ndarray& a) {
  ndarray x = a.astype(kFloat64).ascontiguousarray();
  ndarray mant(x.shape(), kFloat64, Order::C), expo(x.shape(), kInt64, Order::C);
  const double* p = x.size() ? x.typed_data<double>() : nullptr;
  double* pm = mant.size() ? mant.typed_data<double>() : nullptr;
  int64_t* pe = expo.size() ? expo.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < x.size(); ++i) { int e; pm[i] = std::frexp(p[i], &e); pe[i] = e; }
  return {mant, expo};
}

ndarray ldexp(const ndarray& m, const ndarray& e) {
  return binary_double(m, e, [](double mm, double ee) { return std::ldexp(mm, static_cast<int>(ee)); });
}

std::pair<ndarray, ndarray> divmod(const ndarray& a, const ndarray& b) {
  Shape sh;
  auto [af, bf] = broadcast_pair(a, b, sh);
  ndarray q(sh, kFloat64, Order::C), r(sh, kFloat64, Order::C);
  const double* pa = af.size() ? af.typed_data<double>() : nullptr;
  const double* pb = bf.size() ? bf.typed_data<double>() : nullptr;
  double* pq = q.size() ? q.typed_data<double>() : nullptr;
  double* pr = r.size() ? r.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < q.size(); ++i) {
    const double fq = std::floor(pa[i] / pb[i]);
    pq[i] = fq;
    pr[i] = pa[i] - fq * pb[i];  // numpy mod: sign follows divisor
  }
  return {q, r};
}

ndarray unwrap(const ndarray& p) {
  ndarray x = p.astype(kFloat64).ravel().copy();
  const int64_t n = x.size();
  ndarray out(x.shape(), kFloat64, Order::C);
  const double* px = n ? x.typed_data<double>() : nullptr;
  double* o = n ? out.typed_data<double>() : nullptr;
  if (n == 0) return out;
  o[0] = px[0];
  double correction = 0.0;
  for (int64_t i = 1; i < n; ++i) {
    const double d = px[i] - px[i - 1];
    double ddmod = std::fmod(d + M_PI, 2.0 * M_PI);
    if (ddmod < 0) ddmod += 2.0 * M_PI;
    ddmod -= M_PI;
    if (ddmod == -M_PI && d > 0) ddmod = M_PI;
    double ph = ddmod - d;
    if (std::abs(d) < M_PI) ph = 0.0;
    correction += ph;
    o[i] = px[i] + correction;
  }
  return out;
}

ndarray i0(const ndarray& a) { return unary_double(a, i0_scalar); }
ndarray nextafter(const ndarray& a, const ndarray& b) {
  return binary_double(a, b, [](double x, double y) { return std::nextafter(x, y); });
}
ndarray spacing(const ndarray& a) {
  return unary_double(a, [](double x) {
    // numpy: distance to the adjacent value in the direction away from zero
    // (toward +inf for x >= 0, toward -inf for x < 0).
    const double dir = std::copysign(std::numeric_limits<double>::infinity(), x);
    return std::nextafter(x, dir) - x;
  });
}

}  // namespace numpp
