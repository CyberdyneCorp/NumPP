#include "numpp/manip/manip.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/umath/ufunc.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// ===== joining =====
namespace numpp {

ndarray concatenate(const std::vector<ndarray>& arrays, int64_t axis) {
  if (arrays.empty()) throw value_error("need at least one array to concatenate");
  const int64_t nd = arrays.front().ndim();
  const int64_t ax = normalize_axis(axis, nd);
  // Result dtype = promotion over all inputs.
  DType dt = arrays.front().dtype();
  for (size_t i = 1; i < arrays.size(); ++i) dt = result_type(dt, arrays[i].dtype());
  // Output shape: input shape with the concat axis summed.
  Shape out_shape = arrays.front().shape();
  int64_t total = 0;
  for (const auto& arr : arrays) {
    if (arr.ndim() != nd)
      throw value_error("all input arrays must have the same number of dimensions");
    total += arr.shape()[ax];
  }
  out_shape[ax] = total;
  ndarray out(out_shape, dt);
  // Copy each input into its destination slab along the concat axis.
  int64_t pos = 0;
  for (const auto& arr : arrays) {
    const int64_t n = arr.shape()[ax];
    std::vector<IndexItem> items;
    items.reserve(static_cast<size_t>(nd));
    for (int64_t d = 0; d < nd; ++d) {
      if (d == ax) items.push_back(Slice{pos, pos + n, 1});
      else items.push_back(Slice{std::nullopt, std::nullopt, 1});
    }
    ndarray dst = out.index(items);
    copyto(dst, arr);  // casts to out dtype, broadcasts to slab shape
    pos += n;
  }
  return out;
}

ndarray stack(const std::vector<ndarray>& arrays, int64_t axis) {
  if (arrays.empty()) throw value_error("need at least one array to stack");
  // A new dimension is inserted, so the valid axis range is [-(nd+1), nd].
  const int64_t ax = normalize_axis(axis, arrays.front().ndim() + 1);
  std::vector<ndarray> expanded;
  expanded.reserve(arrays.size());
  for (const auto& arr : arrays) expanded.push_back(arr.expand_dims(ax));
  return concatenate(expanded, ax);
}

ndarray hstack(const std::vector<ndarray>& arrays) {
  if (arrays.empty()) throw value_error("need at least one array to hstack");
  std::vector<ndarray> arrs;
  arrs.reserve(arrays.size());
  for (const auto& a : arrays) arrs.push_back(atleast_1d(a));
  // 1-D inputs concatenate along axis 0; higher-D along axis 1.
  const int64_t ax = arrs.front().ndim() == 1 ? 0 : 1;
  return concatenate(arrs, ax);
}

ndarray vstack(const std::vector<ndarray>& arrays) {
  if (arrays.empty()) throw value_error("need at least one array to vstack");
  std::vector<ndarray> arrs;
  arrs.reserve(arrays.size());
  for (const auto& a : arrays) arrs.push_back(atleast_2d(a));
  return concatenate(arrs, 0);
}

ndarray dstack(const std::vector<ndarray>& arrays) {
  if (arrays.empty()) throw value_error("need at least one array to dstack");
  std::vector<ndarray> arrs;
  arrs.reserve(arrays.size());
  for (const auto& a : arrays) arrs.push_back(atleast_3d(a));
  return concatenate(arrs, 2);
}

ndarray column_stack(const std::vector<ndarray>& arrays) {
  if (arrays.empty()) throw value_error("need at least one array to column_stack");
  std::vector<ndarray> arrs;
  arrs.reserve(arrays.size());
  for (const auto& a : arrays) {
    // 1-D inputs become single columns (n -> n x 1); others kept as 2-D.
    if (a.ndim() == 1) arrs.push_back(a.reshape({a.shape()[0], 1}));
    else arrs.push_back(atleast_2d(a));
  }
  return concatenate(arrs, 1);
}

}  // namespace numpp

// ===== splitting =====
namespace numpp {

// Build an index spec that selects [start, stop) along `axis` and the full
// extent on every other dimension (a default Slice{} == ":" full slice).
static std::vector<IndexItem> split_axis_slice(int64_t ndim, int64_t axis,
                                               int64_t start, int64_t stop) {
  std::vector<IndexItem> items(static_cast<size_t>(ndim), Slice{});
  items[static_cast<size_t>(axis)] = Slice{start, stop, 1};
  return items;
}

// array_split: like split but the last sections may be shorter. NumPy gives the
// first (Ntotal % sections) sections one extra element each.
std::vector<ndarray> array_split(const ndarray& a, int64_t sections, int64_t axis) {
  if (sections <= 0)
    throw value_error("number sections must be larger than 0.");
  int64_t ax = normalize_axis(axis, a.ndim());
  int64_t total = a.shape()[static_cast<size_t>(ax)];
  int64_t each = total / sections;
  int64_t extras = total % sections;
  std::vector<ndarray> out;
  out.reserve(static_cast<size_t>(sections));
  int64_t start = 0;
  for (int64_t i = 0; i < sections; ++i) {
    int64_t stop = start + each + (i < extras ? 1 : 0);
    out.push_back(a.index(split_axis_slice(a.ndim(), ax, start, stop)).copy());
    start = stop;
  }
  return out;
}

// split: requires the axis length to divide evenly, otherwise value_error.
std::vector<ndarray> split(const ndarray& a, int64_t sections, int64_t axis) {
  if (sections <= 0)
    throw value_error("number sections must be larger than 0.");
  int64_t ax = normalize_axis(axis, a.ndim());
  if (a.shape()[static_cast<size_t>(ax)] % sections != 0)
    throw value_error("array split does not result in an equal division");
  return array_split(a, sections, axis);
}

// hsplit: split along axis 1 (axis 0 for 1-D inputs).
std::vector<ndarray> hsplit(const ndarray& a, int64_t sections) {
  if (a.ndim() == 0)
    throw value_error("hsplit only works on arrays of 1 or more dimensions");
  return split(a, sections, a.ndim() == 1 ? 0 : 1);
}

// vsplit: split along axis 0 (requires at least 2-D).
std::vector<ndarray> vsplit(const ndarray& a, int64_t sections) {
  if (a.ndim() < 2)
    throw value_error("vsplit only works on arrays of 2 or more dimensions");
  return split(a, sections, 0);
}

}  // namespace numpp

// ===== tiling =====
namespace numpp {

ndarray tile(const ndarray& a, const std::vector<int64_t>& reps) {
  const int64_t n = a.ndim();
  const int64_t d = static_cast<int64_t>(reps.size());
  const int64_t nd = std::max(n, d);

  // Promote the array shape to `nd` dims by prepending 1s.
  Shape pshape(nd, 1);
  for (int64_t i = 0; i < n; ++i) pshape[nd - n + i] = a.shape()[i];

  // Promote reps to length `nd` by prepending 1s.
  std::vector<int64_t> preps(nd, 1);
  for (int64_t i = 0; i < d; ++i) preps[nd - d + i] = reps[i];

  ndarray result = a.ascontiguousarray().reshape(pshape);
  for (int64_t axis = 0; axis < nd; ++axis) {
    if (preps[axis] == 1) continue;
    std::vector<ndarray> parts(static_cast<size_t>(preps[axis]), result);
    result = concatenate(parts, axis);
  }
  return result;
}

ndarray repeat(const ndarray& a, int64_t repeats, std::optional<int64_t> axis) {
  ndarray arr = a;
  int64_t ax = 0;
  if (!axis) {
    arr = a.ravel();  // flatten to 1-D
  } else {
    ax = normalize_axis(*axis, a.ndim());
  }

  const int64_t n = arr.shape()[ax];
  Shape oshape = arr.shape();
  oshape[ax] = n * repeats;
  ndarray out(oshape, arr.dtype());

  for (int64_t j = 0; j < n; ++j) {
    std::vector<IndexItem> src_idx(static_cast<size_t>(arr.ndim()), Slice{});
    src_idx[ax] = Slice{j, j + 1, 1};
    ndarray src = arr.index(src_idx);
    for (int64_t r = 0; r < repeats; ++r) {
      const int64_t pos = j * repeats + r;
      std::vector<IndexItem> dst_idx(static_cast<size_t>(out.ndim()), Slice{});
      dst_idx[ax] = Slice{pos, pos + 1, 1};
      ndarray dst = out.index(dst_idx);
      copyto(dst, src);
    }
  }
  return out;
}

}  // namespace numpp

// ===== rearranging =====
namespace numpp {

// flip: with axis=nullopt flip every axis; otherwise flip the given (possibly
// negative) axis. Implemented as a reversed-stride view that is then copied.
ndarray flip(const ndarray& a, std::optional<int64_t> axis) {
  const int64_t n = a.ndim();
  std::vector<bool> flipped(static_cast<size_t>(n), false);
  if (axis) {
    flipped[static_cast<size_t>(normalize_axis(*axis, n))] = true;
  } else {
    for (int64_t i = 0; i < n; ++i) flipped[static_cast<size_t>(i)] = true;
  }
  std::vector<IndexItem> items;
  items.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    if (flipped[static_cast<size_t>(i)])
      items.push_back(IndexItem{Slice{std::nullopt, std::nullopt, -1}});
    else
      items.push_back(IndexItem{Slice{}});
  }
  return a.index(items).copy();
}

ndarray fliplr(const ndarray& a) { return flip(a, 1); }
ndarray flipud(const ndarray& a) { return flip(a, 0); }

ndarray roll(const ndarray& a, int64_t shift, std::optional<int64_t> axis) {
  // axis=nullopt: flatten (C order), roll, restore the original shape.
  if (!axis) {
    if (a.size() == 0) return a.copy();
    ndarray flat = a.ravel().copy();
    return roll(flat, shift, 0).reshape(a.shape());
  }
  const int64_t n = a.ndim();
  const int64_t ax = normalize_axis(*axis, n);
  const int64_t L = a.shape()[static_cast<size_t>(ax)];
  ndarray out(a.shape(), a.dtype());
  if (L == 0) { copyto(out, a); return out; }
  const int64_t s = ((shift % L) + L) % L;
  if (s == 0) { copyto(out, a); return out; }

  auto full = [&]() { return std::vector<IndexItem>(static_cast<size_t>(n), IndexItem{Slice{}}); };
  // out[s:]   <- a[:L-s]
  {
    auto di = full(); di[static_cast<size_t>(ax)] = IndexItem{Slice{s, std::nullopt, 1}};
    auto si = full(); si[static_cast<size_t>(ax)] = IndexItem{Slice{std::nullopt, L - s, 1}};
    ndarray dst = out.index(di);
    copyto(dst, a.index(si));
  }
  // out[:s]   <- a[L-s:]
  {
    auto di = full(); di[static_cast<size_t>(ax)] = IndexItem{Slice{std::nullopt, s, 1}};
    auto si = full(); si[static_cast<size_t>(ax)] = IndexItem{Slice{L - s, std::nullopt, 1}};
    ndarray dst = out.index(di);
    copyto(dst, a.index(si));
  }
  return out;
}

// rot90: rotate counter-clockwise in the (axis0, axis1) plane. A single 90 deg
// CCW rotation == flip(axis 1) then swapaxes(0,1); apply k (mod 4) times.
ndarray rot90(const ndarray& a, int64_t k) {
  const int64_t kk = ((k % 4) + 4) % 4;
  ndarray r = a.copy();
  for (int64_t i = 0; i < kk; ++i) {
    r = flip(r, 1);              // already a copy
    r = r.swapaxes(0, 1).copy();
  }
  return r;
}

ndarray moveaxis(const ndarray& a, int64_t source, int64_t destination) {
  const int64_t n = a.ndim();
  const int64_t src = normalize_axis(source, n);
  const int64_t dst = normalize_axis(destination, n);
  std::vector<int64_t> order;
  order.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    if (i != src) order.push_back(i);
  order.insert(order.begin() + dst, src);
  return a.transpose(order).copy();
}

ndarray atleast_1d(const ndarray& a) {
  if (a.ndim() >= 1) return a;
  return a.reshape({1});
}

ndarray atleast_2d(const ndarray& a) {
  if (a.ndim() == 0) return a.reshape({1, 1});
  if (a.ndim() == 1) return a.reshape({1, a.shape()[0]});
  return a;
}

ndarray atleast_3d(const ndarray& a) {
  if (a.ndim() == 0) return a.reshape({1, 1, 1});
  if (a.ndim() == 1) return a.reshape({1, a.shape()[0], 1});
  if (a.ndim() == 2) return a.reshape({a.shape()[0], a.shape()[1], 1});
  return a;
}

}  // namespace numpp

// ===== editing =====
namespace numpp {

// Build an index list of full slices on every axis except `axis`, which gets `s`.
static std::vector<IndexItem> editing_axis_slices(int64_t ndim, int64_t axis, Slice s) {
  std::vector<IndexItem> items(static_cast<size_t>(ndim), IndexItem{Slice{}});
  items[static_cast<size_t>(axis)] = IndexItem{s};
  return items;
}

ndarray append(const ndarray& a, const ndarray& values, std::optional<int64_t> axis) {
  // axis None: flatten both operands, then concatenate as 1-D (numpy semantics).
  if (!axis) return concatenate({a.ravel(), values.ravel()}, 0);
  return concatenate({a, values}, *axis);
}

ndarray insert(const ndarray& a, int64_t index, const ndarray& values,
               std::optional<int64_t> axis) {
  // axis None: operate on the flattened array, inserting before `index`.
  if (!axis) {
    ndarray flat = a.ravel();
    int64_t N = flat.size();
    if (index < 0) index += N;
    ndarray v = values.ravel().astype(a.dtype());
    int64_t numnew = v.size();
    ndarray out(Shape{N + numnew}, a.dtype());
    if (index > 0) {
      auto d = out.index({IndexItem{Slice{0, index}}});
      auto s = flat.index({IndexItem{Slice{0, index}}});
      copyto(d, s);
    }
    {
      auto d = out.index({IndexItem{Slice{index, index + numnew}}});
      copyto(d, v);
    }
    if (index < N) {
      auto d = out.index({IndexItem{Slice{index + numnew, std::nullopt}}});
      auto s = flat.index({IndexItem{Slice{index, std::nullopt}}});
      copyto(d, s);
    }
    return out;
  }

  const int64_t nd = a.ndim();
  int64_t ax = normalize_axis(*axis, nd);
  int64_t dimlen = a.shape()[static_cast<size_t>(ax)];
  if (index < 0) index += dimlen;

  // Reproduce numpy's scalar-index broadcasting: give `values` `nd` dims (padding
  // leading 1s), then move its axis 0 to `ax`; the resulting axis length is the
  // number of slices inserted.
  ndarray v = values.astype(a.dtype());
  Shape vs = v.shape();
  while (static_cast<int64_t>(vs.size()) < nd) vs.insert(vs.begin(), 1);
  v = v.reshape(vs);
  std::vector<int64_t> perm;
  for (int64_t i = 1; i < nd; ++i) perm.push_back(i);
  perm.insert(perm.begin() + ax, 0);
  v = v.transpose(perm);
  int64_t numnew = v.shape()[static_cast<size_t>(ax)];

  Shape ns = a.shape();
  ns[static_cast<size_t>(ax)] += numnew;
  ndarray out(ns, a.dtype());
  if (index > 0) {
    auto d = out.index(editing_axis_slices(nd, ax, Slice{0, index}));
    auto s = a.index(editing_axis_slices(nd, ax, Slice{0, index}));
    copyto(d, s);
  }
  {
    auto d = out.index(editing_axis_slices(nd, ax, Slice{index, index + numnew}));
    copyto(d, v);  // broadcasts size-1 dims to the destination slab
  }
  if (index < dimlen) {
    auto d = out.index(editing_axis_slices(nd, ax, Slice{index + numnew, std::nullopt}));
    auto s = a.index(editing_axis_slices(nd, ax, Slice{index, std::nullopt}));
    copyto(d, s);
  }
  return out;
}

ndarray delete_(const ndarray& a, int64_t index, std::optional<int64_t> axis) {
  if (!axis) {
    ndarray flat = a.ravel();
    int64_t N = flat.size();
    if (index < 0) index += N;
    ndarray out(Shape{N - 1}, a.dtype());
    if (index > 0) {
      auto d = out.index({IndexItem{Slice{0, index}}});
      auto s = flat.index({IndexItem{Slice{0, index}}});
      copyto(d, s);
    }
    if (index < N - 1) {
      auto d = out.index({IndexItem{Slice{index, std::nullopt}}});
      auto s = flat.index({IndexItem{Slice{index + 1, std::nullopt}}});
      copyto(d, s);
    }
    return out;
  }

  const int64_t nd = a.ndim();
  int64_t ax = normalize_axis(*axis, nd);
  int64_t dimlen = a.shape()[static_cast<size_t>(ax)];
  if (index < 0) index += dimlen;
  Shape ns = a.shape();
  ns[static_cast<size_t>(ax)] -= 1;
  ndarray out(ns, a.dtype());
  if (index > 0) {
    auto d = out.index(editing_axis_slices(nd, ax, Slice{0, index}));
    auto s = a.index(editing_axis_slices(nd, ax, Slice{0, index}));
    copyto(d, s);
  }
  if (index < dimlen - 1) {
    auto d = out.index(editing_axis_slices(nd, ax, Slice{index, std::nullopt}));
    auto s = a.index(editing_axis_slices(nd, ax, Slice{index + 1, std::nullopt}));
    copyto(d, s);
  }
  return out;
}

ndarray resize(const ndarray& a, const Shape& new_shape) {
  // numpy.resize: cyclically repeat the flattened data to fill new_shape.
  int64_t total = 1;
  for (int64_t d : new_shape) total *= d;
  ndarray out(new_shape, a.dtype());
  if (total == 0) return out;
  ndarray flat = a.ravel().ascontiguousarray();
  int64_t N = flat.size();
  if (N == 0) return zeros(new_shape, a.dtype());  // numpy fills empty source with 0
  int64_t isz = a.itemsize();
  char* dst = out.bytes();
  const char* src = flat.bytes();
  for (int64_t i = 0; i < total; ++i)
    std::memcpy(dst + i * isz, src + (i % N) * isz, static_cast<size_t>(isz));
  return out;
}

ndarray pad(const ndarray& a, const std::vector<std::pair<int64_t, int64_t>>& pad_width,
            const std::string& mode, double constant_value) {
  const int64_t nd = a.ndim();
  std::vector<std::pair<int64_t, int64_t>> pw = pad_width;
  if (static_cast<int64_t>(pw.size()) == 1 && nd > 1)
    pw.assign(static_cast<size_t>(nd), pad_width[0]);  // broadcast a single (before,after)

  Shape ns(static_cast<size_t>(nd));
  for (int64_t i = 0; i < nd; ++i) {
    auto u = static_cast<size_t>(i);
    ns[u] = a.shape()[u] + pw[u].first + pw[u].second;
  }
  ndarray out = (mode == "constant") ? full(ns, constant_value, a.dtype())
                                     : ndarray(ns, a.dtype());

  // Place the original array in the center slab.
  std::vector<IndexItem> center(static_cast<size_t>(nd));
  for (int64_t i = 0; i < nd; ++i) {
    auto u = static_cast<size_t>(i);
    center[u] = IndexItem{Slice{pw[u].first, pw[u].first + a.shape()[u]}};
  }
  {
    auto d = out.index(center);
    copyto(d, a);
  }
  if (mode == "constant") return out;

  if (mode == "edge") {
    // Fill borders axis by axis. Each pass copies the nearest valid edge slice
    // across the full extent of the other axes; sequential passes resolve corners.
    for (int64_t k = 0; k < nd; ++k) {
      auto u = static_cast<size_t>(k);
      int64_t before = pw[u].first;
      int64_t orig = a.shape()[u];
      if (orig <= 0) continue;
      int64_t last = before + orig - 1;
      if (before > 0) {
        auto s = out.index(editing_axis_slices(nd, k, Slice{before, before + 1}));
        auto d = out.index(editing_axis_slices(nd, k, Slice{0, before}));
        copyto(d, s);
      }
      if (pw[u].second > 0) {
        auto s = out.index(editing_axis_slices(nd, k, Slice{last, last + 1}));
        auto d = out.index(editing_axis_slices(nd, k, Slice{last + 1, std::nullopt}));
        copyto(d, s);
      }
    }
    return out;
  }
  throw value_error("pad: unsupported mode (expected 'constant' or 'edge')");
}

}  // namespace numpp
