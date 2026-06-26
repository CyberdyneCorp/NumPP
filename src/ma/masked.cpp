#include "numpp/ma/masked.hpp"

#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <vector>
#include <optional>


namespace numpp {

namespace ma {

namespace {

// Flatten an array to a contiguous 1-D C-order copy of float64.
inline ndarray flat_f64(const ndarray& a) {
  return a.astype(kFloat64).ravel(Order::C).ascontiguousarray();
}
// Flatten an array to a contiguous 1-D C-order copy of bool.
inline ndarray flat_bool(const ndarray& a) {
  return a.astype(kBool).ravel(Order::C).ascontiguousarray();
}

// Build a 0-d float64 array holding a scalar value.
inline ndarray scalar0d(double v) {
  ndarray r(Shape{}, kFloat64);
  r.set_item<double>({}, v);
  return r;
}

}  // namespace

MaskedArray masked_array(const ndarray& data, const ndarray& mask) {
  return MaskedArray{data.astype(kFloat64).copy(), mask.astype(kBool).copy()};
}

MaskedArray masked_where(const ndarray& condition, const ndarray& a) {
  return MaskedArray{a.astype(kFloat64).copy(), condition.astype(kBool).copy()};
}

MaskedArray masked_invalid(const ndarray& a) {
  ndarray data = a.astype(kFloat64).copy();
  ndarray flat = flat_f64(data);
  const int64_t n = flat.size();
  const double* p = flat.typed_data<double>();
  ndarray mask(data.shape(), kBool);
  ndarray mflat = mask.ravel(Order::C);  // view, contiguous (mask owns C-order)
  bool* mp = mflat.typed_data<bool>();
  for (int64_t i = 0; i < n; ++i) {
    double v = p[i];
    mp[i] = std::isnan(v) || std::isinf(v);
  }
  return MaskedArray{data, mask};
}

MaskedArray masked_equal(const ndarray& a, double value) {
  ndarray data = a.astype(kFloat64).copy();
  ndarray flat = flat_f64(data);
  const int64_t n = flat.size();
  const double* p = flat.typed_data<double>();
  ndarray mask(data.shape(), kBool);
  ndarray mflat = mask.ravel(Order::C);
  bool* mp = mflat.typed_data<bool>();
  for (int64_t i = 0; i < n; ++i) mp[i] = (p[i] == value);
  return MaskedArray{data, mask};
}

MaskedArray masked_greater(const ndarray& a, double value) {
  ndarray data = a.astype(kFloat64).copy();
  ndarray flat = flat_f64(data);
  const int64_t n = flat.size();
  const double* p = flat.typed_data<double>();
  ndarray mask(data.shape(), kBool);
  ndarray mflat = mask.ravel(Order::C);
  bool* mp = mflat.typed_data<bool>();
  for (int64_t i = 0; i < n; ++i) mp[i] = (p[i] > value);
  return MaskedArray{data, mask};
}

ndarray filled(const MaskedArray& m, double fill_value) {
  ndarray out = m.data.astype(kFloat64).copy();
  ndarray oflat = out.ravel(Order::C);
  double* op = oflat.typed_data<double>();
  ndarray mflat = flat_bool(m.mask);
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = oflat.size();
  for (int64_t i = 0; i < n; ++i)
    if (mp[i]) op[i] = fill_value;
  return out;
}

ndarray compressed(const MaskedArray& m) {
  ndarray dflat = flat_f64(m.data);
  ndarray mflat = flat_bool(m.mask);
  const double* dp = dflat.typed_data<double>();
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = dflat.size();
  int64_t k = 0;
  for (int64_t i = 0; i < n; ++i)
    if (!mp[i]) ++k;
  ndarray out(Shape{k}, kFloat64);
  double* op = out.typed_data<double>();
  int64_t j = 0;
  for (int64_t i = 0; i < n; ++i)
    if (!mp[i]) op[j++] = dp[i];
  return out;
}

int64_t count(const MaskedArray& m) {
  ndarray mflat = flat_bool(m.mask);
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = mflat.size();
  int64_t c = 0;
  for (int64_t i = 0; i < n; ++i)
    if (!mp[i]) ++c;
  return c;
}

ndarray sum(const MaskedArray& m) {
  ndarray dflat = flat_f64(m.data);
  ndarray mflat = flat_bool(m.mask);
  const double* dp = dflat.typed_data<double>();
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = dflat.size();
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i)
    if (!mp[i]) s += dp[i];
  return scalar0d(s);
}

ndarray mean(const MaskedArray& m) {
  ndarray dflat = flat_f64(m.data);
  ndarray mflat = flat_bool(m.mask);
  const double* dp = dflat.typed_data<double>();
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = dflat.size();
  double s = 0.0;
  int64_t c = 0;
  for (int64_t i = 0; i < n; ++i)
    if (!mp[i]) { s += dp[i]; ++c; }
  return scalar0d(c > 0 ? s / static_cast<double>(c) : std::nan(""));
}

ndarray min(const MaskedArray& m) {
  ndarray dflat = flat_f64(m.data);
  ndarray mflat = flat_bool(m.mask);
  const double* dp = dflat.typed_data<double>();
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = dflat.size();
  double best = 0.0;
  bool any = false;
  for (int64_t i = 0; i < n; ++i) {
    if (mp[i]) continue;
    if (!any || dp[i] < best) { best = dp[i]; any = true; }
  }
  return scalar0d(any ? best : std::nan(""));
}

ndarray max(const MaskedArray& m) {
  ndarray dflat = flat_f64(m.data);
  ndarray mflat = flat_bool(m.mask);
  const double* dp = dflat.typed_data<double>();
  const bool* mp = mflat.typed_data<bool>();
  const int64_t n = dflat.size();
  double best = 0.0;
  bool any = false;
  for (int64_t i = 0; i < n; ++i) {
    if (mp[i]) continue;
    if (!any || dp[i] > best) { best = dp[i]; any = true; }
  }
  return scalar0d(any ? best : std::nan(""));
}

}  // namespace ma

}  // namespace numpp
