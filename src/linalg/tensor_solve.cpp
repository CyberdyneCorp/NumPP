#include <algorithm>
#include <vector>

#include "numpp/core/error.hpp"
#include "numpp/linalg/linalg.hpp"

namespace numpp {
namespace linalg {

ndarray tensorsolve(const ndarray& a_in, const ndarray& b, const std::vector<int64_t>& axes) {
  ndarray a = a_in;
  const int64_t an = a.ndim();
  if (!axes.empty()) {
    std::vector<int64_t> allaxes;
    for (int64_t i = 0; i < an; ++i) allaxes.push_back(i);
    for (int64_t k : axes) {  // move each named axis to the end (numpy order)
      allaxes.erase(std::remove(allaxes.begin(), allaxes.end(), k), allaxes.end());
      allaxes.push_back(k);
    }
    a = a.transpose(allaxes);
  }
  const int64_t bn = b.ndim();
  if (an < bn) throw linalg_error("tensorsolve: a.ndim must be >= b.ndim");
  // The trailing (an - bn) axes of a form the unknown's shape and must square
  // against the leading axes (which match b).
  Shape oldshape(a.shape().end() - (an - bn), a.shape().end());
  int64_t prod = 1;
  for (int64_t d : oldshape) prod *= d;
  if (a.size() != prod * prod)
    throw linalg_error("tensorsolve: prod(a.shape[b.ndim:]) must equal prod(a.shape[:b.ndim])");
  ndarray A = a.ascontiguousarray().reshape({prod, prod});
  ndarray B = b.ascontiguousarray().reshape({prod});
  return solve(A, B).reshape(oldshape);
}

ndarray tensorinv(const ndarray& a, int64_t ind) {
  const int64_t nd = a.ndim();
  if (ind <= 0 || ind > nd) throw value_error("Invalid ind argument");
  const Shape& sh = a.shape();
  int64_t prod = 1;  // product of the trailing axes
  for (int64_t i = ind; i < nd; ++i) prod *= sh[i];
  int64_t lead = 1;  // product of the leading axes
  for (int64_t i = 0; i < ind; ++i) lead *= sh[i];
  Shape invshape;  // numpy: shape[ind:] + shape[:ind]
  for (int64_t i = ind; i < nd; ++i) invshape.push_back(sh[i]);
  for (int64_t i = 0; i < ind; ++i) invshape.push_back(sh[i]);
  ndarray A = a.ascontiguousarray().reshape({prod, lead});
  return inv(A).reshape(invshape);  // inv enforces prod == lead (square)
}

}  // namespace linalg
}  // namespace numpp
