#include "numpp/core/creation.hpp"

#include <cmath>
#include <cstring>

namespace numpp {
namespace {

// Write `value` (as a double/complex source) into every element of a fresh array.
ndarray filled(const Shape& shape, DType dtype, double value) {
  ndarray a(shape, dtype, Order::C);
  visit_dtype(dtype.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    T v;
    if constexpr (std::is_same_v<T, half>) v = half(static_cast<float>(value));
    else if constexpr (std::is_same_v<T, std::complex<float>>) v = std::complex<float>(static_cast<float>(value), 0);
    else if constexpr (std::is_same_v<T, std::complex<double>>) v = std::complex<double>(value, 0);
    else if constexpr (std::is_same_v<T, bool>) v = value != 0.0;
    else v = static_cast<T>(value);
    a.fill<T>(v);
  });
  return a;
}

}  // namespace

ndarray empty(const Shape& shape, DType dtype) { return ndarray(shape, dtype, Order::C); }
ndarray zeros(const Shape& shape, DType dtype) { return filled(shape, dtype, 0.0); }
ndarray ones(const Shape& shape, DType dtype) { return filled(shape, dtype, 1.0); }
ndarray full(const Shape& shape, double value, DType dtype) { return filled(shape, dtype, value); }

ndarray empty_like(const ndarray& a) { return empty(a.shape(), a.dtype()); }
ndarray zeros_like(const ndarray& a) { return zeros(a.shape(), a.dtype()); }
ndarray ones_like(const ndarray& a) { return ones(a.shape(), a.dtype()); }
ndarray full_like(const ndarray& a, double value) { return full(a.shape(), value, a.dtype()); }

ndarray eye(int64_t n, int64_t m, int64_t k, DType dtype) {
  if (m < 0) m = n;
  ndarray a = zeros({n, m}, dtype);
  for (int64_t i = 0; i < n; ++i) {
    int64_t j = i + k;
    if (j >= 0 && j < m) {
      visit_dtype(dtype.id(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        T one;
        if constexpr (std::is_same_v<T, half>) one = half(1.0f);
        else if constexpr (std::is_same_v<T, std::complex<float>>) one = {1.0f, 0};
        else if constexpr (std::is_same_v<T, std::complex<double>>) one = {1.0, 0};
        else one = static_cast<T>(1);
        a.set_item<T>({i, j}, one);
      });
    }
  }
  return a;
}

ndarray identity(int64_t n, DType dtype) { return eye(n, n, 0, dtype); }

ndarray arange(double start, double stop, double step, DType dtype) {
  if (step == 0.0) throw value_error("arange step cannot be zero");
  int64_t count = static_cast<int64_t>(std::ceil((stop - start) / step));
  if (count < 0) count = 0;
  ndarray a(Shape{count}, dtype, Order::C);
  visit_dtype(dtype.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    for (int64_t i = 0; i < count; ++i) {
      double val = start + static_cast<double>(i) * step;
      T v;
      if constexpr (std::is_same_v<T, half>) v = half(static_cast<float>(val));
      else if constexpr (std::is_same_v<T, std::complex<float>>) v = std::complex<float>(static_cast<float>(val), 0);
      else if constexpr (std::is_same_v<T, std::complex<double>>) v = std::complex<double>(val, 0);
      else if constexpr (std::is_same_v<T, bool>) v = val != 0.0;
      else v = static_cast<T>(val);
      a.set_item<T>({i}, v);
    }
  });
  return a;
}

ndarray arange(double start, double stop, double step) { return arange(start, stop, step, default_int()); }
ndarray arange(double stop) { return arange(0.0, stop, 1.0, default_int()); }

ndarray linspace(double start, double stop, int64_t num, bool endpoint, DType dtype) {
  if (num < 0) throw value_error("number of samples must be non-negative");
  ndarray a(Shape{num}, dtype, Order::C);
  const int64_t div = endpoint ? num - 1 : num;
  const double step = div > 0 ? (stop - start) / static_cast<double>(div) : 0.0;
  visit_dtype(dtype.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    for (int64_t i = 0; i < num; ++i) {
      double val = (endpoint && i == num - 1) ? stop : start + static_cast<double>(i) * step;
      T v;
      if constexpr (std::is_same_v<T, half>) v = half(static_cast<float>(val));
      else if constexpr (std::is_same_v<T, std::complex<float>>) v = std::complex<float>(static_cast<float>(val), 0);
      else if constexpr (std::is_same_v<T, std::complex<double>>) v = std::complex<double>(val, 0);
      else v = static_cast<T>(val);
      a.set_item<T>({i}, v);
    }
  });
  return a;
}

}  // namespace numpp
