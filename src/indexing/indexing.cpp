#include "numpp/indexing/indexing.hpp"

#include "numpp/core/creation.hpp"   // zeros/full
#include "numpp/core/shape.hpp"      // normalize_axis
#include "numpp/manip/manip.hpp"     // concatenate
#include "numpp/umath/ufunc.hpp"     // nonzero

#include <cstring>
#include <vector>

namespace numpp {
namespace {

// Odometer increment of a C-order multi-index over `shape`. Returns false at wrap.
bool next_index(std::vector<int64_t>& idx, const Shape& shape) {
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    if (++idx[d] < shape[d]) return true;
    idx[d] = 0;
  }
  return false;
}

// Read indices array as int64 values (flattened, C order), normalizing negatives against `n`.
std::vector<int64_t> index_values(const ndarray& indices, int64_t n) {
  ndarray idx = indices.astype(kInt64).ascontiguousarray();
  const int64_t cnt = idx.size();
  const int64_t* p = cnt ? idx.typed_data<int64_t>() : nullptr;
  std::vector<int64_t> out(cnt);
  for (int64_t i = 0; i < cnt; ++i) {
    int64_t k = p[i];
    if (k < 0) k += n;
    if (k < 0 || k >= n) throw index_error("index out of bounds");
    out[i] = k;
  }
  return out;
}

}  // namespace

ndarray take(const ndarray& a, const ndarray& indices, std::optional<int64_t> axis) {
  const int64_t isz = a.dtype().itemsize();
  if (!axis) {
    ndarray src = a.ascontiguousarray().reshape({a.size()});
    std::vector<int64_t> idx = index_values(indices, a.size());
    ndarray out(indices.shape(), a.dtype(), Order::C);
    const char* s = src.bytes();
    char* o = out.size() ? out.bytes() : nullptr;
    for (size_t n = 0; n < idx.size(); ++n) std::memcpy(o + n * isz, s + idx[n] * isz, isz);
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  std::vector<int64_t> idx = index_values(indices, a.shape()[ax]);
  std::vector<ndarray> slabs;
  slabs.reserve(idx.size());
  for (int64_t k : idx) {
    std::vector<IndexItem> sel;
    for (int64_t d = 0; d < a.ndim(); ++d) sel.push_back(d == ax ? IndexItem{k} : IndexItem{Slice{}});
    slabs.push_back(a.index(sel).expand_dims(ax).copy());
  }
  return concatenate(slabs, ax);
}

ndarray take_along_axis(const ndarray& a, const ndarray& indices, int64_t axis) {
  const int64_t ax = normalize_axis(axis, a.ndim());
  const int64_t isz = a.dtype().itemsize();
  ndarray idx = indices.astype(kInt64);
  const Shape osh = idx.shape();
  ndarray out(osh, a.dtype(), Order::C);
  if (out.size() == 0) return out;
  std::vector<int64_t> oi(osh.size(), 0);
  do {
    int64_t k = idx.item<int64_t>(oi);
    if (k < 0) k += a.shape()[ax];
    std::vector<int64_t> ai = oi;
    ai[ax] = k;
    std::memcpy(out.element_ptr(oi), a.element_ptr(ai), isz);
  } while (next_index(oi, osh));
  return out;
}

void put(ndarray& a, const ndarray& indices, const ndarray& values) {
  const int64_t n = a.size();
  const int64_t isz = a.dtype().itemsize();
  std::vector<int64_t> idx = index_values(indices, n);
  ndarray vals = values.astype(a.dtype()).ascontiguousarray().reshape({values.size()});
  const int64_t vn = vals.size();
  const char* v = vn ? vals.bytes() : nullptr;
  char* dst = a.bytes();  // C-order flat; a is treated as a flat buffer like numpy.put
  for (size_t j = 0; j < idx.size(); ++j)
    std::memcpy(dst + idx[j] * isz, v + (static_cast<int64_t>(j) % vn) * isz, isz);
}

ndarray diagonal(const ndarray& a, int64_t offset, int64_t axis1, int64_t axis2) {
  const int64_t a1 = normalize_axis(axis1, a.ndim()), a2 = normalize_axis(axis2, a.ndim());
  ndarray m = a;
  if (!(a1 == a.ndim() - 2 && a2 == a.ndim() - 1)) m = a.swapaxes(a1, a.ndim() - 2).swapaxes(a2, a.ndim() - 1);
  if (m.ndim() != 2) throw value_error("diagonal: only 2-D (or two chosen axes) supported");
  const int64_t rows = m.shape()[0], cols = m.shape()[1];
  int64_t len = offset >= 0 ? std::min(rows, cols - offset) : std::min(rows + offset, cols);
  if (len < 0) len = 0;
  const int64_t isz = a.dtype().itemsize();
  ndarray out({len}, a.dtype(), Order::C);
  for (int64_t i = 0; i < len; ++i) {
    const int64_t r = offset >= 0 ? i : i - offset;
    const int64_t c = offset >= 0 ? i + offset : i;
    std::memcpy(out.element_ptr({i}), m.element_ptr({r, c}), isz);
  }
  return out;
}

ndarray argwhere(const ndarray& a) {
  std::vector<ndarray> nz = nonzero(a);
  const int64_t nd = static_cast<int64_t>(nz.size());
  const int64_t N = nd ? nz[0].size() : 0;
  ndarray out({N, nd}, kInt64, Order::C);
  for (int64_t d = 0; d < nd; ++d) {
    ndarray col = nz[d].astype(kInt64).ascontiguousarray();
    for (int64_t i = 0; i < N; ++i) out.set_item<int64_t>({i, d}, col.item<int64_t>({i}));
  }
  return out;
}

ndarray extract(const ndarray& condition, const ndarray& a) {
  ndarray cond = condition.astype(kBool).ascontiguousarray().reshape({condition.size()});
  ndarray flat = a.ascontiguousarray().reshape({a.size()});
  const int64_t isz = a.dtype().itemsize();
  const bool* c = cond.size() ? cond.typed_data<bool>() : nullptr;
  const int64_t m = std::min(cond.size(), flat.size());
  std::vector<int64_t> keep;
  for (int64_t i = 0; i < m; ++i) if (c[i]) keep.push_back(i);
  ndarray out({static_cast<int64_t>(keep.size())}, a.dtype(), Order::C);
  const char* s = flat.bytes();
  char* o = out.size() ? out.bytes() : nullptr;
  for (size_t n = 0; n < keep.size(); ++n) std::memcpy(o + n * isz, s + keep[n] * isz, isz);
  return out;
}

ndarray compress(const ndarray& condition, const ndarray& a, std::optional<int64_t> axis) {
  if (!axis) return extract(condition, a);
  const int64_t ax = normalize_axis(*axis, a.ndim());
  ndarray cond = condition.astype(kBool).ascontiguousarray();
  const bool* c = cond.size() ? cond.typed_data<bool>() : nullptr;
  std::vector<int64_t> keep;
  for (int64_t i = 0; i < cond.size(); ++i) if (c[i]) keep.push_back(i);
  ndarray idx({static_cast<int64_t>(keep.size())}, kInt64, Order::C);
  for (size_t i = 0; i < keep.size(); ++i) idx.set_item<int64_t>({static_cast<int64_t>(i)}, keep[i]);
  return take(a, idx, ax);
}

ndarray choose(const ndarray& indices, const std::vector<ndarray>& choices) {
  if (choices.empty()) throw value_error("choose: empty choices");
  ndarray idx = indices.astype(kInt64);
  const DType dt = choices[0].dtype();
  const int64_t isz = dt.itemsize();
  const Shape osh = idx.shape();
  ndarray out(osh, dt, Order::C);
  if (out.size() == 0) return out;
  std::vector<ndarray> ch;
  ch.reserve(choices.size());
  for (const auto& c : choices) ch.push_back(c.astype(dt).broadcast_to(osh));
  std::vector<int64_t> oi(osh.size(), 0);
  do {
    int64_t k = idx.item<int64_t>(oi);
    if (k < 0) k += static_cast<int64_t>(choices.size());
    std::memcpy(out.element_ptr(oi), ch[k].element_ptr(oi), isz);
  } while (next_index(oi, osh));
  return out;
}

ndarray select(const std::vector<ndarray>& condlist, const std::vector<ndarray>& choicelist,
               double default_value) {
  if (condlist.empty() || condlist.size() != choicelist.size())
    throw value_error("select: condlist/choicelist size mismatch");
  const Shape osh = choicelist[0].shape();
  ndarray out = full(osh, default_value, kFloat64);
  std::vector<char> written(out.size(), 0);
  for (size_t c = 0; c < condlist.size(); ++c) {
    ndarray cond = condlist[c].astype(kBool).broadcast_to(osh);
    ndarray choice = choicelist[c].astype(kFloat64).broadcast_to(osh);
    std::vector<int64_t> oi(osh.size(), 0);
    int64_t f = 0;
    do {
      if (!written[f] && cond.item<bool>(oi)) {
        out.set_item<double>(oi, choice.item<double>(oi));
        written[f] = 1;
      }
      ++f;
    } while (next_index(oi, osh));
  }
  return out;
}

ndarray ravel_multi_index(const std::vector<ndarray>& multi_index, const Shape& dims) {
  const int64_t nd = static_cast<int64_t>(dims.size());
  if (static_cast<int64_t>(multi_index.size()) != nd)
    throw value_error("ravel_multi_index: coordinate count != dims");
  std::vector<ndarray> coords;
  coords.reserve(nd);
  for (const auto& mi : multi_index) coords.push_back(mi.astype(kInt64).ascontiguousarray());
  const int64_t N = nd ? coords[0].size() : 0;
  ndarray out({N}, kInt64, Order::C);
  for (int64_t i = 0; i < N; ++i) {
    int64_t flat = 0;
    for (int64_t d = 0; d < nd; ++d) flat = flat * dims[d] + coords[d].item<int64_t>({i});
    out.set_item<int64_t>({i}, flat);
  }
  return out;
}

std::vector<ndarray> unravel_index(const ndarray& indices, const Shape& dims) {
  const int64_t nd = static_cast<int64_t>(dims.size());
  ndarray idx = indices.astype(kInt64).ascontiguousarray();
  const int64_t N = idx.size();
  std::vector<ndarray> out;
  out.reserve(nd);
  for (int64_t d = 0; d < nd; ++d) out.push_back(ndarray(idx.shape(), kInt64, Order::C));
  for (int64_t i = 0; i < N; ++i) {
    int64_t rem = idx.item<int64_t>({i});
    for (int64_t d = nd - 1; d >= 0; --d) {
      out[d].set_item<int64_t>({i}, dims[d] ? rem % dims[d] : 0);
      if (dims[d]) rem /= dims[d];
    }
  }
  return out;
}

}  // namespace numpp
