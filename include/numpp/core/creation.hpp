#pragma once

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

#include <cstdint>
#include <type_traits>

namespace numpp {

NUMPP_API ndarray empty(const Shape& shape, DType dtype = kFloat64);
NUMPP_API ndarray zeros(const Shape& shape, DType dtype = kFloat64);
NUMPP_API ndarray ones(const Shape& shape, DType dtype = kFloat64);
NUMPP_API ndarray full(const Shape& shape, double value, DType dtype = kFloat64);

NUMPP_API ndarray empty_like(const ndarray& a);
NUMPP_API ndarray zeros_like(const ndarray& a);
NUMPP_API ndarray ones_like(const ndarray& a);
NUMPP_API ndarray full_like(const ndarray& a, double value);

NUMPP_API ndarray eye(int64_t n, int64_t m = -1, int64_t k = 0, DType dtype = kFloat64);
NUMPP_API ndarray identity(int64_t n, DType dtype = kFloat64);

// arange: the argument type drives the dtype, matching numpy — integer arguments
// give the platform integer dtype, floating arguments give float64. Pass an
// explicit DType to override. arange(stop) ranges [0, stop).
NUMPP_API ndarray arange(double start, double stop, double step, DType dtype);
NUMPP_API ndarray arange(double start, double stop, double step = 1.0);  // -> float64
NUMPP_API ndarray arange(double stop);                                   // -> float64
NUMPP_API ndarray arange(int64_t start, int64_t stop, int64_t step = 1); // -> int64
NUMPP_API ndarray arange(int64_t stop);                                  // -> int64

// Forward other integral literals (e.g. plain `int`) to the int64 overloads via
// an exact template match, so arange(6) resolves to the integer path instead of
// being ambiguous between the int64 and double overloads.
template <class I, std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, bool> &&
                                        !std::is_same_v<I, int64_t>, int> = 0>
inline ndarray arange(I stop) { return arange(static_cast<int64_t>(stop)); }
template <class A, class B, class C = int,
          std::enable_if_t<std::is_integral_v<A> && std::is_integral_v<B> && std::is_integral_v<C> &&
                               !(std::is_same_v<A, int64_t> && std::is_same_v<B, int64_t> &&
                                 std::is_same_v<C, int64_t>),
                           int> = 0>
inline ndarray arange(A start, B stop, C step = 1) {
  return arange(static_cast<int64_t>(start), static_cast<int64_t>(stop), static_cast<int64_t>(step));
}

NUMPP_API ndarray linspace(double start, double stop, int64_t num = 50, bool endpoint = true,
                           DType dtype = kFloat64);

}  // namespace numpp
