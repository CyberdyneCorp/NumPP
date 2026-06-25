#pragma once

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

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

// arange: default dtype is the platform default integer (pass a float dtype for
// fractional steps). arange(stop) ranges [0, stop).
NUMPP_API ndarray arange(double start, double stop, double step, DType dtype);
NUMPP_API ndarray arange(double start, double stop, double step = 1.0);
NUMPP_API ndarray arange(double stop);

NUMPP_API ndarray linspace(double start, double stop, int64_t num = 50, bool endpoint = true,
                           DType dtype = kFloat64);

}  // namespace numpp
