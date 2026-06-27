#include "numpp/lib/memory_overlap.hpp"

#include <cstdint>
#include <cstdlib>

namespace numpp {

namespace {

// Inclusive byte interval [lo, hi] occupied by the array within its Buffer.
// Returns false for a zero-size array (no bytes occupied -> never overlaps).
bool byte_interval(const ndarray& a, int64_t& lo, int64_t& hi) {
  if (a.size() == 0) return false;
  lo = a.offset();
  int64_t span = 0;
  const Shape& shp = a.shape();
  const Strides& str = a.strides();
  for (int64_t i = 0; i < a.ndim(); ++i) {
    span += (shp[i] - 1) * std::llabs(str[i]);
  }
  hi = a.offset() + span + a.itemsize() - 1;
  return true;
}

bool intervals_overlap(const ndarray& a, const ndarray& b) {
  // Different underlying buffers can never share memory in this model.
  if (a.buffer().get() != b.buffer().get()) return false;
  if (a.buffer().get() == nullptr) return false;
  int64_t alo, ahi, blo, bhi;
  if (!byte_interval(a, alo, ahi)) return false;
  if (!byte_interval(b, blo, bhi)) return false;
  return alo <= bhi && blo <= ahi;
}

}  // namespace

bool may_share_memory(const ndarray& a, const ndarray& b) {
  return intervals_overlap(a, b);
}

bool shares_memory(const ndarray& a, const ndarray& b) {
  // Faithful bounds-based check: identical to may_share_memory for arrays over
  // the same buffer (NumPp does not run the full Diophantine solver NumPy uses).
  return intervals_overlap(a, b);
}

}  // namespace numpp
