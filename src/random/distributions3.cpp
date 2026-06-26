#include "numpp/random/distributions3.hpp"

#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <cstdint>
#include <vector>


namespace numpp {

namespace random {
namespace {

// Noncentral chi-square with `df` degrees of freedom and noncentrality `nonc`.
double noncentral_chisquare_scalar(Generator& g, double df, double nonc) {
  if (df > 1.0) {
    const double n = g.next_gauss() + std::sqrt(nonc);
    return n * n + 2.0 * g.next_std_gamma((df - 1.0) * 0.5);
  }
  // df <= 1: mix in a Poisson count of extra unit chi-squares.
  // Simple fallback adequate for statistical use.
  const double n = g.next_gauss() + std::sqrt(nonc);
  return n * n;
}

// Central chi-square with `k` degrees of freedom.
double chisquare_scalar(Generator& g, double k) {
  return 2.0 * g.next_std_gamma(k * 0.5);
}

}  // namespace

ndarray hypergeometric(Generator& g, int64_t ngood, int64_t nbad, int64_t nsample,
                       const Shape& size) {
  ndarray out(size, kInt64, Order::C);
  int64_t* p = out.size() ? out.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) {
    int64_t rg = ngood, rb = nbad, cnt = 0;
    for (int64_t s = 0; s < nsample; ++s) {
      const int64_t total = rg + rb;
      if (total <= 0) break;
      if (g.next_double() * static_cast<double>(total) < static_cast<double>(rg)) {
        ++cnt;
        --rg;
      } else {
        --rb;
      }
    }
    p[i] = cnt;
  }
  return out;
}

ndarray noncentral_f(Generator& g, double dfnum, double dfden, double nonc,
                     const Shape& size) {
  ndarray out(size, kFloat64, Order::C);
  double* p = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) {
    const double num = noncentral_chisquare_scalar(g, dfnum, nonc) / dfnum;
    const double den = chisquare_scalar(g, dfden) / dfden;
    p[i] = num / den;
  }
  return out;
}

ndarray bytes(Generator& g, int64_t length) {
  ndarray out(Shape{length}, kUInt8, Order::C);
  uint8_t* p = out.size() ? out.typed_data<uint8_t>() : nullptr;
  int64_t i = 0;
  while (i < length) {
    uint64_t w = g.next_raw();
    for (int b = 0; b < 8 && i < length; ++b, ++i) {
      p[i] = static_cast<uint8_t>(w & 0xFFu);
      w >>= 8;
    }
  }
  return out;
}

}  // namespace random

}  // namespace numpp
