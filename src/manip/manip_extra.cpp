#include "numpp/manip/manip_extra.hpp"

#include "numpp/manip/manip.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <string>
#include <vector>


namespace numpp {

// ===== manip extras: block, dsplit, trim_zeros, rollaxis =====

ndarray block(const std::vector<std::vector<ndarray>>& blocks) {
  if (blocks.empty()) throw value_error("block: need at least one row of blocks");
  // Concatenate each row along the last axis, then stack the rows along the
  // second-to-last axis (numpy.block, restricted to 2-level nesting).
  std::vector<ndarray> rows;
  rows.reserve(blocks.size());
  for (const auto& row : blocks) {
    if (row.empty()) throw value_error("block: empty row of blocks");
    const int64_t nd = row.front().ndim();
    rows.push_back(concatenate(row, nd - 1));
  }
  const int64_t nd = rows.front().ndim();
  return concatenate(rows, nd - 2);
}

std::vector<ndarray> dsplit(const ndarray& a, int64_t sections) {
  if (a.ndim() < 3)
    throw value_error("dsplit only works on arrays of 3 or more dimensions");
  return array_split(a, sections, 2);
}

ndarray trim_zeros(const ndarray& a, const std::string& trim) {
  if (a.ndim() != 1) throw value_error("trim_zeros: array must be 1-D");
  const int64_t n = a.size();
  ndarray f = a.astype(kFloat64);
  const bool front = trim.find('f') != std::string::npos || trim.find('F') != std::string::npos;
  const bool back = trim.find('b') != std::string::npos || trim.find('B') != std::string::npos;

  int64_t lo = 0;
  if (front) {
    while (lo < n && f.item<double>({lo}) == 0.0) ++lo;
  }
  int64_t hi = n - 1;
  if (back) {
    while (hi >= lo && f.item<double>({hi}) == 0.0) --hi;
  }
  if (lo > hi) return a.index({IndexItem{Slice{int64_t{0}, int64_t{0}, 1}}}).copy();
  return a.index({IndexItem{Slice{lo, hi + 1, 1}}}).copy();
}

ndarray rollaxis(const ndarray& a, int64_t axis, int64_t start) {
  const int64_t n = a.ndim();
  const int64_t ax = normalize_axis(axis, n);
  if (start < 0) start += n;
  if (start < 0 || start > n)
    throw value_error("rollaxis: start argument out of range");
  if (ax < start) start -= 1;
  if (ax == start) return a.copy();

  std::vector<int64_t> perm;
  perm.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    if (i != ax) perm.push_back(i);
  perm.insert(perm.begin() + start, ax);
  return a.transpose(perm).copy();
}

}  // namespace numpp
