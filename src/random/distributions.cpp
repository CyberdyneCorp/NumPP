#include "numpp/random/distributions.hpp"

#include "numpp/umath/ufunc.hpp"   // exp

#include <cmath>
#include <functional>

namespace numpp {
namespace random {
namespace {

// Draw a uniform [0,1) array and map each value through f.
ndarray uniform_transform(Generator& g, const Shape& size, const std::function<double(double)>& f) {
  ndarray u = g.random(size).ascontiguousarray();
  ndarray out(u.shape(), kFloat64, Order::C);
  const double* pu = u.size() ? u.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < u.size(); ++i) o[i] = f(pu[i]);
  return out;
}

}  // namespace

ndarray laplace(Generator& g, double loc, double scale, const Shape& size) {
  return uniform_transform(g, size, [=](double u) {
    const double d = u - 0.5;
    return d <= 0.0 ? loc + scale * std::log(2.0 * u) : loc - scale * std::log(2.0 * (1.0 - u));
  });
}

ndarray logistic(Generator& g, double loc, double scale, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return loc + scale * std::log(u / (1.0 - u)); });
}

ndarray gumbel(Generator& g, double loc, double scale, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return loc - scale * std::log(-std::log(u)); });
}

ndarray rayleigh(Generator& g, double scale, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return scale * std::sqrt(-2.0 * std::log(1.0 - u)); });
}

ndarray weibull(Generator& g, double a, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return std::pow(-std::log(1.0 - u), 1.0 / a); });
}

ndarray pareto(Generator& g, double a, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return std::pow(1.0 - u, -1.0 / a) - 1.0; });
}

ndarray power(Generator& g, double a, const Shape& size) {
  return uniform_transform(g, size, [=](double u) { return std::pow(u, 1.0 / a); });
}

ndarray standard_cauchy(Generator& g, const Shape& size) {
  return uniform_transform(g, size, [](double u) { return std::tan(M_PI * (u - 0.5)); });
}

ndarray triangular(Generator& g, double left, double mode, double right, const Shape& size) {
  const double base = right - left;
  const double fc = base > 0 ? (mode - left) / base : 0.0;
  return uniform_transform(g, size, [=](double u) {
    if (u < fc) return left + std::sqrt(u * base * (mode - left));
    return right - std::sqrt((1.0 - u) * base * (right - mode));
  });
}

ndarray lognormal(Generator& g, double mean, double sigma, const Shape& size) {
  return exp(g.normal(mean, sigma, size));
}

}  // namespace random
}  // namespace numpp
