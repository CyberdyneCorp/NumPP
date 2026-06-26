#include "numpp/polynomial/orthopoly_calc.hpp"

#include "numpp/polynomial/polynomial.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <vector>


namespace numpp {

namespace polynomial {
namespace {

// --- 1-D float64 array <-> vector helpers (lowest-first coeffs) ---
std::vector<double> di_to_vec(const ndarray& a) {
  ndarray f = a.astype(kFloat64).ravel().copy();
  const int64_t n = f.size();
  const double* p = n ? f.typed_data<double>() : nullptr;
  return std::vector<double>(p, p + n);
}
ndarray di_from_vec(const std::vector<double>& v) {
  ndarray o({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) o.typed_data<double>()[i] = v[i];
  return o;
}

// Evaluate a basis series (coeffs tmp, lowest-first) at scalar x = 0 using the
// already-shipped *val(x, c) routines.
double di_val0(ndarray (*valfn)(const ndarray&, const ndarray&),
               const std::vector<double>& tmp) {
  ndarray x({1}, kFloat64, Order::C);
  x.typed_data<double>()[0] = 0.0;
  ndarray r = valfn(x, di_from_vec(tmp));
  return r.item<double>({0});
}

// ---- single-step derivatives (len n -> len n-1, or {0} if n<=1) ----
std::vector<double> hermder1(std::vector<double> c) {
  const int64_t n = static_cast<int64_t>(c.size());
  if (n <= 1) return {0.0};
  std::vector<double> der(n - 1, 0.0);
  for (int64_t j = 1; j < n; ++j) der[j - 1] = 2.0 * j * c[j];
  return der;
}
std::vector<double> hermeder1(std::vector<double> c) {
  const int64_t n = static_cast<int64_t>(c.size());
  if (n <= 1) return {0.0};
  std::vector<double> der(n - 1, 0.0);
  for (int64_t j = 1; j < n; ++j) der[j - 1] = static_cast<double>(j) * c[j];
  return der;
}
std::vector<double> chebder1(std::vector<double> c) {
  const int64_t n = static_cast<int64_t>(c.size());
  if (n <= 1) return {0.0};
  std::vector<double> der(n - 1, 0.0);
  for (int64_t j = n - 1; j >= 3; --j) {
    der[j - 1] = 2.0 * j * c[j];
    c[j - 2] += (static_cast<double>(j) * c[j]) / static_cast<double>(j - 2);
  }
  if (n > 2) der[1] = 4.0 * c[2];
  der[0] = c[1];
  return der;
}
std::vector<double> legder1(std::vector<double> c) {
  const int64_t n = static_cast<int64_t>(c.size());
  if (n <= 1) return {0.0};
  std::vector<double> der(n - 1, 0.0);
  for (int64_t j = n - 1; j >= 3; --j) {
    der[j - 1] = (2.0 * j - 1.0) * c[j];
    c[j - 2] += c[j];
  }
  if (n > 2) der[1] = 3.0 * c[2];
  der[0] = c[1];
  return der;
}
std::vector<double> lagder1(std::vector<double> c) {
  const int64_t n = static_cast<int64_t>(c.size());
  if (n <= 1) return {0.0};
  std::vector<double> der(n - 1, 0.0);
  for (int64_t j = n - 1; j >= 1; --j) {
    der[j - 1] = -c[j];
    c[j - 1] += c[j];
  }
  return der;
}

// ---- single-step integrals (len n -> len n+1, then constant correction) ----
std::vector<double> hermint1(const std::vector<double>& c, double k) {
  const int64_t n = static_cast<int64_t>(c.size());
  std::vector<double> tmp(n + 1, 0.0);
  for (int64_t j = 0; j < n; ++j) tmp[j + 1] = c[j] / (2.0 * (j + 1));
  tmp[0] += k - di_val0(hermval, tmp);
  return tmp;
}
std::vector<double> hermeint1(const std::vector<double>& c, double k) {
  const int64_t n = static_cast<int64_t>(c.size());
  std::vector<double> tmp(n + 1, 0.0);
  for (int64_t j = 0; j < n; ++j) tmp[j + 1] = c[j] / static_cast<double>(j + 1);
  tmp[0] += k - di_val0(hermeval, tmp);
  return tmp;
}
std::vector<double> chebint1(const std::vector<double>& c, double k) {
  const int64_t n = static_cast<int64_t>(c.size());
  std::vector<double> tmp(n + 1, 0.0);
  tmp[1] += c[0];
  if (n > 1) tmp[2] += c[1] / 4.0;
  for (int64_t j = 2; j < n; ++j) {
    tmp[j + 1] += c[j] / (2.0 * (j + 1));
    tmp[j - 1] -= c[j] / (2.0 * (j - 1));
  }
  tmp[0] += k - di_val0(chebval, tmp);
  return tmp;
}
std::vector<double> legint1(const std::vector<double>& c, double k) {
  const int64_t n = static_cast<int64_t>(c.size());
  std::vector<double> tmp(n + 1, 0.0);
  tmp[1] += c[0];
  for (int64_t j = 1; j < n; ++j) {
    tmp[j + 1] += c[j] / (2.0 * j + 1.0);
    tmp[j - 1] -= c[j] / (2.0 * j + 1.0);
  }
  tmp[0] += k - di_val0(legval, tmp);
  return tmp;
}
std::vector<double> lagint1(const std::vector<double>& c, double k) {
  const int64_t n = static_cast<int64_t>(c.size());
  std::vector<double> tmp(n + 1, 0.0);
  tmp[0] += c[0];
  tmp[1] -= c[0];
  for (int64_t j = 1; j < n; ++j) {
    tmp[j] += c[j];
    tmp[j + 1] -= c[j];
  }
  tmp[0] += k - di_val0(lagval, tmp);
  return tmp;
}

// Apply a single-step derivative m times.
ndarray di_der(const ndarray& c, int64_t m, std::vector<double> (*step)(std::vector<double>)) {
  if (m < 0) value_error("The order of derivative must be non-negative");
  std::vector<double> v = di_to_vec(c);
  for (int64_t i = 0; i < m; ++i) v = step(v);
  return di_from_vec(v);
}
// Apply a single-step integral m times. numpy uses k as the list of integration
// constants padded with zeros, so a scalar k applies on the first integration
// only and 0 on the rest.
ndarray di_int(const ndarray& c, int64_t m, double k,
               std::vector<double> (*step)(const std::vector<double>&, double)) {
  if (m < 0) value_error("The order of integration must be non-negative");
  std::vector<double> v = di_to_vec(c);
  for (int64_t i = 0; i < m; ++i) v = step(v, i == 0 ? k : 0.0);
  return di_from_vec(v);
}

}  // namespace

ndarray chebder(const ndarray& c, int64_t m) { return di_der(c, m, chebder1); }
ndarray chebint(const ndarray& c, int64_t m, double k) { return di_int(c, m, k, chebint1); }
ndarray legder(const ndarray& c, int64_t m) { return di_der(c, m, legder1); }
ndarray legint(const ndarray& c, int64_t m, double k) { return di_int(c, m, k, legint1); }
ndarray hermder(const ndarray& c, int64_t m) { return di_der(c, m, hermder1); }
ndarray hermint(const ndarray& c, int64_t m, double k) { return di_int(c, m, k, hermint1); }
ndarray hermeder(const ndarray& c, int64_t m) { return di_der(c, m, hermeder1); }
ndarray hermeint(const ndarray& c, int64_t m, double k) { return di_int(c, m, k, hermeint1); }
ndarray lagder(const ndarray& c, int64_t m) { return di_der(c, m, lagder1); }
ndarray lagint(const ndarray& c, int64_t m, double k) { return di_int(c, m, k, lagint1); }

}  // namespace polynomial

}  // namespace numpp
