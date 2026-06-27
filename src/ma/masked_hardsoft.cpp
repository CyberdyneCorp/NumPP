#include "numpp/ma/masked.hpp"

#include "numpp/core/error.hpp"

namespace numpp {
namespace ma {
namespace {

// Write `v` into `a` at `idx`, converting to the array's dtype.
void write_value(ndarray& a, const std::vector<int64_t>& idx, double v) {
  const DType dt = a.dtype();
  if (dt == kFloat64) a.set_item<double>(idx, v);
  else if (dt == kFloat32) a.set_item<float>(idx, static_cast<float>(v));
  else if (dt == kInt64) a.set_item<int64_t>(idx, static_cast<int64_t>(v));
  else if (dt == kInt32) a.set_item<int32_t>(idx, static_cast<int32_t>(v));
  else if (dt == kInt16) a.set_item<int16_t>(idx, static_cast<int16_t>(v));
  else if (dt == kInt8) a.set_item<int8_t>(idx, static_cast<int8_t>(v));
  else if (dt == kBool) a.set_item<bool>(idx, v != 0.0);
  else throw type_error("masked setitem: unsupported data dtype");
}

}  // namespace

void harden_mask(MaskedArray& m) { m.hard = true; }
void soften_mask(MaskedArray& m) { m.hard = false; }
bool hardmask(const MaskedArray& m) { return m.hard; }

void setitem(MaskedArray& m, const std::vector<int64_t>& idx, double value) {
  const bool masked_here = m.mask.item<bool>(idx);
  // A hard mask protects a masked position: assignment is a no-op.
  if (m.hard && masked_here) return;
  write_value(m.data, idx, value);
  // Soft mask (or assigning to an unmasked hard-mask cell): unmask the position.
  m.mask.set_item<bool>(idx, false);
}

}  // namespace ma
}  // namespace numpp
