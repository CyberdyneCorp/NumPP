#include "numpp/sorting/sorting_extra.hpp"

#include "numpp/indexing/indexing.hpp"   // take
#include "numpp/sorting/sorting.hpp"     // searchsorted (base)

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

namespace numpp {

ndarray lexsort(const std::vector<ndarray>& keys) {
  if (keys.empty()) throw value_error("lexsort: need at least one key");
  const int64_t n = keys.front().size();
  std::vector<std::vector<double>> kv;
  kv.reserve(keys.size());
  for (const auto& k : keys) {
    ndarray f = k.astype(kFloat64).ravel().copy();
    const double* p = f.size() ? f.typed_data<double>() : nullptr;
    kv.emplace_back(p, p + f.size());
  }
  std::vector<int64_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  // Stable sort by each key from least significant (first) to most significant (last).
  for (const auto& key : kv)
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return key[a] < key[b]; });
  ndarray out({n}, kInt64, Order::C);
  for (int64_t i = 0; i < n; ++i) out.typed_data<int64_t>()[i] = idx[i];
  return out;
}

ndarray sort_complex(const ndarray& a) {
  ndarray c = a.astype(kComplex128).ravel().copy();
  const int64_t n = c.size();
  std::complex<double>* p = n ? c.typed_data<std::complex<double>>() : nullptr;
  std::stable_sort(p, p + n, [](const std::complex<double>& x, const std::complex<double>& y) {
    if (x.real() != y.real()) return x.real() < y.real();
    return x.imag() < y.imag();
  });
  return c;
}

ndarray searchsorted(const ndarray& a, const ndarray& v, const std::string& side, const ndarray& sorter) {
  return searchsorted(take(a, sorter), v, side);
}

}  // namespace numpp
