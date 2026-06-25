#include "numpp/core/ndarray.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>

namespace numpp {
namespace {

// Widen any supported scalar to complex<double> (real types -> imag 0).
template <class S>
std::complex<double> widen(S s) {
  if constexpr (std::is_same_v<S, std::complex<float>>) return {s.real(), s.imag()};
  else if constexpr (std::is_same_v<S, std::complex<double>>) return s;
  else if constexpr (std::is_same_v<S, half>) return {static_cast<double>(static_cast<float>(s)), 0.0};
  else if constexpr (std::is_same_v<S, bool>) return {s ? 1.0 : 0.0, 0.0};
  else return {static_cast<double>(s), 0.0};
}

template <class D>
D narrow(std::complex<double> c) {
  if constexpr (std::is_same_v<D, std::complex<float>>)
    return {static_cast<float>(c.real()), static_cast<float>(c.imag())};
  else if constexpr (std::is_same_v<D, std::complex<double>>) return c;
  else if constexpr (std::is_same_v<D, half>) return half(static_cast<float>(c.real()));
  else if constexpr (std::is_same_v<D, bool>) return c.real() != 0.0 || c.imag() != 0.0;
  else return static_cast<D>(c.real());
}

// Exact integer->integer casts go direct; everything else through complex<double>.
template <class Dst, class Src>
Dst cast_one(Src s) {
  if constexpr (std::is_integral_v<Src> && std::is_integral_v<Dst>) return static_cast<Dst>(s);
  else return narrow<Dst>(widen<Src>(s));
}

// CPython slice semantics -> (first index, count, step).
struct SliceResult { int64_t start, count, step; };
SliceResult resolve_slice(int64_t length, const Slice& sl) {
  const int64_t step = sl.step;
  if (step == 0) throw value_error("slice step cannot be zero");
  const int64_t lower = step < 0 ? -1 : 0;
  const int64_t upper = step < 0 ? length - 1 : length;
  auto clamp = [&](int64_t v) { return std::min(std::max(v, lower), upper); };
  int64_t start = sl.start ? (*sl.start < 0 ? *sl.start + length : *sl.start)
                           : (step < 0 ? upper : lower);
  start = clamp(start);
  int64_t stop = sl.stop ? (*sl.stop < 0 ? *sl.stop + length : *sl.stop)
                         : (step < 0 ? lower : upper);
  stop = clamp(stop);
  int64_t count = 0;
  if (step < 0) { if (stop < start) count = (start - stop - 1) / (-step) + 1; }
  else { if (start < stop) count = (stop - start - 1) / step + 1; }
  return {start, count, step};
}

}  // namespace

ndarray::ndarray(Shape shape, DType dtype, Order order)
    : dtype_(dtype), shape_(std::move(shape)) {
  strides_ = contiguous_strides(shape_, dtype.itemsize(), order);
  buffer_ = std::make_shared<Buffer>(std::max<int64_t>(size() * itemsize(), 1));
  offset_ = 0;
  writeable_ = true;
}

ndarray::ndarray(std::shared_ptr<Buffer> buffer, DType dtype, Shape shape, Strides strides,
                 int64_t offset, bool writeable)
    : buffer_(std::move(buffer)), dtype_(dtype), shape_(std::move(shape)),
      strides_(std::move(strides)), offset_(offset), writeable_(writeable) {}

const char* ndarray::element_ptr(const std::vector<int64_t>& idx) const {
  if (static_cast<int64_t>(idx.size()) != ndim())
    throw index_error("wrong number of indices for array");
  int64_t off = offset_;
  for (int64_t i = 0; i < ndim(); ++i) {
    int64_t k = idx[i] < 0 ? idx[i] + shape_[i] : idx[i];
    if (k < 0 || k >= shape_[i])
      throw index_error("index " + std::to_string(idx[i]) + " is out of bounds for axis " +
                        std::to_string(i) + " with size " + std::to_string(shape_[i]));
    off += k * strides_[i];
  }
  return buffer_->data() + off;
}
char* ndarray::element_ptr(const std::vector<int64_t>& idx) {
  return const_cast<char*>(static_cast<const ndarray*>(this)->element_ptr(idx));
}

void ndarray::fill_bytes(const void* scalar) {
  const int64_t es = itemsize();
  for_each_offset([&](int64_t off) { std::memcpy(buffer_->data() + off, scalar, es); });
}

ndarray ndarray::reshape(const Shape& new_shape_in) const {
  Shape new_shape = new_shape_in;
  int64_t neg = -1, known = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(new_shape.size()); ++i) {
    if (new_shape[i] == -1) {
      if (neg != -1) throw value_error("can only specify one unknown dimension");
      neg = i;
    } else {
      known *= new_shape[i];
    }
  }
  if (neg != -1) {
    if (known == 0 || size() % known != 0) throw value_error("cannot reshape array");
    new_shape[neg] = size() / known;
  }
  if (shape_size(new_shape) != size()) throw value_error("cannot reshape array of size " +
                                                         std::to_string(size()));
  if (c_contiguous()) {
    Strides st = contiguous_strides(new_shape, itemsize(), Order::C);
    return ndarray(buffer_, dtype_, new_shape, st, offset_, writeable_);
  }
  ndarray contig = copy(Order::C);
  Strides st = contiguous_strides(new_shape, itemsize(), Order::C);
  return ndarray(contig.buffer_, dtype_, new_shape, st, contig.offset_, true);
}

ndarray ndarray::transpose() const {
  std::vector<int64_t> axes(ndim());
  for (int64_t i = 0; i < ndim(); ++i) axes[i] = ndim() - 1 - i;
  return transpose(axes);
}

ndarray ndarray::transpose(const std::vector<int64_t>& axes) const {
  if (static_cast<int64_t>(axes.size()) != ndim()) throw value_error("axes don't match array");
  Shape ns(ndim());
  Strides nst(ndim());
  std::vector<char> seen(ndim(), 0);
  for (int64_t i = 0; i < ndim(); ++i) {
    int64_t a = normalize_axis(axes[i], ndim());
    if (seen[a]) throw value_error("repeated axis in transpose");
    seen[a] = 1;
    ns[i] = shape_[a];
    nst[i] = strides_[a];
  }
  return ndarray(buffer_, dtype_, ns, nst, offset_, writeable_);
}

ndarray ndarray::swapaxes(int64_t a, int64_t b) const {
  std::vector<int64_t> axes(ndim());
  for (int64_t i = 0; i < ndim(); ++i) axes[i] = i;
  std::swap(axes[normalize_axis(a, ndim())], axes[normalize_axis(b, ndim())]);
  return transpose(axes);
}

ndarray ndarray::squeeze() const {
  Shape ns;
  Strides nst;
  for (int64_t i = 0; i < ndim(); ++i) {
    if (shape_[i] != 1) { ns.push_back(shape_[i]); nst.push_back(strides_[i]); }
  }
  return ndarray(buffer_, dtype_, ns, nst, offset_, writeable_);
}

ndarray ndarray::expand_dims(int64_t axis) const {
  int64_t a = axis < 0 ? axis + ndim() + 1 : axis;
  if (a < 0 || a > ndim()) throw axis_error("axis out of range for expand_dims");
  Shape ns = shape_;
  Strides nst = strides_;
  ns.insert(ns.begin() + a, 1);
  nst.insert(nst.begin() + a, 0);
  return ndarray(buffer_, dtype_, ns, nst, offset_, writeable_);
}

ndarray ndarray::ravel(Order order) const {
  const bool ok = (order == Order::C && c_contiguous()) || (order == Order::F && f_contiguous());
  if (ok) return reshape({size()});
  return flatten(order);
}

ndarray ndarray::flatten(Order order) const {
  ndarray c = copy(order);
  Strides st = {itemsize()};
  return ndarray(c.buffer_, dtype_, {size()}, st, 0, true);
}

ndarray ndarray::index(const std::vector<IndexItem>& items) const {
  Shape ns;
  Strides nst;
  int64_t off = offset_;
  int64_t axis = 0;
  for (const auto& it : items) {
    if (std::holds_alternative<NewAxis>(it)) {
      ns.push_back(1);
      nst.push_back(0);
    } else if (std::holds_alternative<int64_t>(it)) {
      if (axis >= ndim()) throw index_error("too many indices for array");
      int64_t k = std::get<int64_t>(it);
      int64_t kk = k < 0 ? k + shape_[axis] : k;
      if (kk < 0 || kk >= shape_[axis])
        throw index_error("index " + std::to_string(k) + " is out of bounds for axis " +
                          std::to_string(axis));
      off += kk * strides_[axis];
      ++axis;
    } else {
      if (axis >= ndim()) throw index_error("too many indices for array");
      SliceResult r = resolve_slice(shape_[axis], std::get<Slice>(it));
      off += r.start * strides_[axis];
      ns.push_back(r.count);
      nst.push_back(strides_[axis] * r.step);
      ++axis;
    }
  }
  for (; axis < ndim(); ++axis) { ns.push_back(shape_[axis]); nst.push_back(strides_[axis]); }
  return ndarray(buffer_, dtype_, ns, nst, off, writeable_);
}

ndarray ndarray::operator[](int64_t i) const {
  return index({IndexItem{static_cast<int64_t>(i)}});
}

ndarray ndarray::copy(Order order) const {
  ndarray out(shape_, dtype_, order);
  const int64_t es = itemsize();
  // Iterate the same multi-index over source and destination.
  if (size() == 0) return out;
  std::vector<int64_t> idx(ndim(), 0);
  while (true) {
    int64_t soff = offset_, doff = out.offset_;
    for (int64_t i = 0; i < ndim(); ++i) { soff += idx[i] * strides_[i]; doff += idx[i] * out.strides_[i]; }
    std::memcpy(out.buffer_->data() + doff, buffer_->data() + soff, es);
    int64_t ax = ndim() - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < shape_[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
  return out;
}

ndarray ndarray::ascontiguousarray() const {
  if (c_contiguous()) return *this;
  return copy(Order::C);
}
ndarray ndarray::asfortranarray() const {
  if (f_contiguous()) return *this;
  return copy(Order::F);
}

ndarray ndarray::broadcast_to(const Shape& target) const {
  const int64_t nt = static_cast<int64_t>(target.size());
  if (nt < ndim()) throw value_error("cannot broadcast to fewer dimensions");
  Strides nst(nt, 0);
  const int64_t pad = nt - ndim();
  for (int64_t i = 0; i < nt; ++i) {
    if (i < pad) { nst[i] = 0; continue; }
    int64_t sd = shape_[i - pad];
    if (sd == target[i]) nst[i] = strides_[i - pad];
    else if (sd == 1) nst[i] = 0;
    else throw value_error("cannot broadcast shape to requested shape");
  }
  return ndarray(buffer_, dtype_, target, nst, offset_, /*writeable=*/false);
}

ndarray ndarray::astype(DType dt, Casting casting) const {
  if (!can_cast(dtype_, dt, casting))
    throw type_error(std::string("cannot cast from ") + dtype_.name() + " to " + dt.name() +
                     " under the given casting rule");
  ndarray out(shape_, dt, Order::C);
  if (size() == 0) return out;
  const int64_t ses = itemsize(), des = dt.itemsize();
  std::vector<int64_t> idx(ndim(), 0);
  while (true) {
    int64_t soff = offset_, doff = out.offset_;
    for (int64_t i = 0; i < ndim(); ++i) { soff += idx[i] * strides_[i]; doff += idx[i] * out.strides_[i]; }
    const char* sp = buffer_->data() + soff;
    char* dp = out.buffer_->data() + doff;
    visit_dtype(dtype_.id(), [&](auto stag) {
      using S = typename decltype(stag)::type;
      S sv; std::memcpy(&sv, sp, ses);
      visit_dtype(dt.id(), [&](auto dtag) {
        using D = typename decltype(dtag)::type;
        D dv = cast_one<D>(sv);
        std::memcpy(dp, &dv, des);
      });
    });
    int64_t ax = ndim() - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < shape_[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
  return out;
}

bool array_equal(const ndarray& a, const ndarray& b) {
  if (a.shape() != b.shape()) return false;
  DType dt = result_type(a.dtype(), b.dtype());
  ndarray aa = a.astype(dt), bb = b.astype(dt);
  bool eq = true;
  std::vector<int64_t> idx(aa.ndim(), 0);
  if (aa.size() == 0) return true;
  while (eq) {
    int64_t ao = aa.offset(), bo = bb.offset();
    for (int64_t i = 0; i < aa.ndim(); ++i) { ao += idx[i] * aa.strides()[i]; bo += idx[i] * bb.strides()[i]; }
    visit_dtype(dt.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      T av, bv;
      std::memcpy(&av, aa.buffer()->data() + ao, sizeof(T));
      std::memcpy(&bv, bb.buffer()->data() + bo, sizeof(T));
      if (widen(av) != widen(bv)) eq = false;
    });
    int64_t ax = aa.ndim() - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < aa.shape()[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
  return eq;
}

bool allclose(const ndarray& a, const ndarray& b, double rtol, double atol, bool equal_nan) {
  Shape bs = broadcast_shapes(a.shape(), b.shape());
  ndarray aa = a.astype(kComplex128).broadcast_to(bs);
  ndarray bb = b.astype(kComplex128).broadcast_to(bs);
  bool ok = true;
  if (shape_size(bs) == 0) return true;
  std::vector<int64_t> idx(static_cast<size_t>(bs.size()), 0);
  const int64_t nd = static_cast<int64_t>(bs.size());
  while (ok) {
    int64_t ao = aa.offset(), bo = bb.offset();
    for (int64_t i = 0; i < nd; ++i) { ao += idx[i] * aa.strides()[i]; bo += idx[i] * bb.strides()[i]; }
    std::complex<double> av, bv;
    std::memcpy(&av, aa.buffer()->data() + ao, sizeof(av));
    std::memcpy(&bv, bb.buffer()->data() + bo, sizeof(bv));
    auto close = [&](double x, double y) {
      if (std::isnan(x) || std::isnan(y)) return equal_nan && std::isnan(x) && std::isnan(y);
      if (std::isinf(x) || std::isinf(y)) return x == y;
      return std::abs(x - y) <= atol + rtol * std::abs(y);
    };
    if (!close(av.real(), bv.real()) || !close(av.imag(), bv.imag())) ok = false;
    int64_t ax = nd - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < bs[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
  return ok;
}

}  // namespace numpp
