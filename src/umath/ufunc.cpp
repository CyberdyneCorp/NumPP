#include "numpp/umath/ufunc.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace numpp {
namespace {

template <class T>
inline constexpr bool is_cplx =
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

// Compute-dtype proxy: arithmetic never runs on half or bool storage.
DType proxy(DType d) {
  if (d == kFloat16) return kFloat32;
  if (d == kBool) return kInt16;
  return d;
}
DType real_of(DType d) { return d == kComplex64 ? kFloat32 : kFloat64; }
bool is_int_kind(DType d) { return d.kind() == 'i' || d.kind() == 'u' || d == kBool; }
DType to_float(DType d) {  // transcendental result dtype
  if (d.is_complex()) return d;
  if (d == kFloat16 || d == kFloat32 || d == kFloat64) return d;
  return kFloat64;
}

// ---- broadcasting zip helpers (output is C-contiguous) ----
template <class T, class R, class F>
void zip2(const ndarray& aa, const ndarray& bb, ndarray& res, F f) {
  const int64_t nd = res.ndim(), n = res.size();
  if (n == 0) return;
  R* out = res.template typed_data<R>();
  std::vector<int64_t> idx(nd, 0);
  int64_t lin = 0;
  while (true) {
    int64_t ao = aa.offset(), bo = bb.offset();
    for (int64_t i = 0; i < nd; ++i) { ao += idx[i] * aa.strides()[i]; bo += idx[i] * bb.strides()[i]; }
    T av, bv;
    std::memcpy(&av, aa.buffer()->data() + ao, sizeof(T));
    std::memcpy(&bv, bb.buffer()->data() + bo, sizeof(T));
    out[lin++] = f(av, bv);
    int64_t ax = nd - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < res.shape()[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
}
template <class T, class R, class F>
void zip1(const ndarray& aa, ndarray& res, F f) {
  const int64_t nd = res.ndim(), n = res.size();
  if (n == 0) return;
  R* out = res.template typed_data<R>();
  std::vector<int64_t> idx(nd, 0);
  int64_t lin = 0;
  while (true) {
    int64_t ao = aa.offset();
    for (int64_t i = 0; i < nd; ++i) ao += idx[i] * aa.strides()[i];
    T av;
    std::memcpy(&av, aa.buffer()->data() + ao, sizeof(T));
    out[lin++] = f(av);
    int64_t ax = nd - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < res.shape()[ax]) break; idx[ax] = 0; }
    if (ax < 0) break;
  }
}

// ---- scalar kernels ----
enum class BinOp { Add, Sub, Mul, Div, FloorDiv, Mod, Power, Min, Max, And, Or, Xor };

template <class T>
T int_floordiv(T x, T y) {
  if (y == 0) return 0;                       // numpy: int //0 -> 0 (with warning)
  T q = x / y, r = x % y;
  if (r != 0 && ((r < 0) != (y < 0))) --q;    // round toward -inf
  return q;
}

template <class T>
T bin_scalar(BinOp op, T x, T y) {
  if constexpr (is_cplx<T>) {
    switch (op) {
      case BinOp::Add: return x + y;
      case BinOp::Sub: return x - y;
      case BinOp::Mul: return x * y;
      case BinOp::Div: return x / y;
      case BinOp::Power: return std::pow(x, y);
      default: throw type_error("operation not supported for complex dtype");
    }
  } else if constexpr (std::is_integral_v<T>) {
    switch (op) {
      case BinOp::Add: return static_cast<T>(x + y);
      case BinOp::Sub: return static_cast<T>(x - y);
      case BinOp::Mul: return static_cast<T>(x * y);
      case BinOp::Div: return static_cast<T>(x / y);
      case BinOp::FloorDiv: return int_floordiv<T>(x, y);
      case BinOp::Mod: return static_cast<T>(x - int_floordiv<T>(x, y) * y);
      case BinOp::Power: return static_cast<T>(std::pow(static_cast<double>(x), static_cast<double>(y)));
      case BinOp::Min: return std::min(x, y);
      case BinOp::Max: return std::max(x, y);
      case BinOp::And: return static_cast<T>(x & y);
      case BinOp::Or: return static_cast<T>(x | y);
      case BinOp::Xor: return static_cast<T>(x ^ y);
    }
  } else {  // real floating
    switch (op) {
      case BinOp::Add: return x + y;
      case BinOp::Sub: return x - y;
      case BinOp::Mul: return x * y;
      case BinOp::Div: return x / y;
      case BinOp::FloorDiv: return std::floor(x / y);
      case BinOp::Mod: { T r = x - std::floor(x / y) * y; return r; }
      case BinOp::Power: return std::pow(x, y);
      case BinOp::Min: return std::fmin(x, y);
      case BinOp::Max: return std::fmax(x, y);
      default: throw type_error("bitwise operation requires integer dtype");
    }
  }
  return T{};
}

enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };
template <class T>
bool cmp_scalar(CmpOp op, T x, T y) {
  if constexpr (is_cplx<T>) {
    switch (op) {
      case CmpOp::Eq: return x == y;
      case CmpOp::Ne: return x != y;
      default: throw type_error("ordering comparison not supported for complex dtype");
    }
  } else {
    switch (op) {
      case CmpOp::Eq: return x == y;
      case CmpOp::Ne: return x != y;
      case CmpOp::Lt: return x < y;
      case CmpOp::Le: return x <= y;
      case CmpOp::Gt: return x > y;
      case CmpOp::Ge: return x >= y;
    }
  }
  return false;
}

// ---- binary dispatch ----
DType arith_out(BinOp op, DType a, DType b) {
  DType p = result_type(a, b);
  switch (op) {
    case BinOp::Sub:
      if (p == kBool) throw type_error("numpy boolean subtract is not supported");
      return p;
    case BinOp::Div:
      return is_int_kind(p) ? kFloat64 : p;
    case BinOp::FloorDiv:
    case BinOp::Mod:
      if (p.is_complex()) throw type_error("floor_divide/remainder not supported for complex");
      return p;
    case BinOp::Min: case BinOp::Max:
      if (p.is_complex()) throw not_implemented_error("minimum/maximum on complex not yet implemented");
      return p;
    case BinOp::And: case BinOp::Or: case BinOp::Xor:
      if (p.is_floating() || p.is_complex()) throw type_error("bitwise op requires integer/bool dtype");
      return p;
    default:
      return p;
  }
}

ndarray binary(const ndarray& a, const ndarray& b, BinOp op) {
  DType out_dt = arith_out(op, a.dtype(), b.dtype());
  DType cdt = proxy(out_dt);
  Shape bs = broadcast_shapes(a.shape(), b.shape());
  ndarray aa = a.astype(cdt).broadcast_to(bs);
  ndarray bb = b.astype(cdt).broadcast_to(bs);
  ndarray res(bs, cdt, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    // half and bool are proxied away before compute; guard so the switch compiles.
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) { (void)res; }
    else zip2<T, T>(aa, bb, res, [&](T x, T y) { return bin_scalar<T>(op, x, y); });
  });
  return cdt == out_dt ? res : res.astype(out_dt);
}

ndarray compare(const ndarray& a, const ndarray& b, CmpOp op) {
  DType cdt = result_type(a.dtype(), b.dtype());
  if (cdt == kFloat16) cdt = kFloat32;
  Shape bs = broadcast_shapes(a.shape(), b.shape());
  ndarray aa = a.astype(cdt).broadcast_to(bs);
  ndarray bb = b.astype(cdt).broadcast_to(bs);
  ndarray res(bs, kBool, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)res; }
    else zip2<T, bool>(aa, bb, res, [&](T x, T y) { return cmp_scalar<T>(op, x, y); });
  });
  return res;
}

ndarray logical(const ndarray& a, const ndarray& b, int op) {  // 0 and,1 or,2 xor
  Shape bs = broadcast_shapes(a.shape(), b.shape());
  ndarray aa = a.astype(kBool).broadcast_to(bs);
  ndarray bb = b.astype(kBool).broadcast_to(bs);
  ndarray res(bs, kBool, Order::C);
  zip2<bool, bool>(aa, bb, res, [&](bool x, bool y) {
    return op == 0 ? (x && y) : op == 1 ? (x || y) : (x != y);
  });
  return res;
}

// ---- unary ----
enum class UnOp { Neg, Pos, Square, Recip, Sign };
ndarray unary_same(const ndarray& a, UnOp op) {
  DType out_dt = a.dtype();
  if ((op == UnOp::Neg || op == UnOp::Pos) && out_dt == kBool)
    throw type_error("negative/positive not supported for boolean dtype");
  DType cdt = proxy(out_dt);
  ndarray aa = a.astype(cdt);
  ndarray res(aa.shape(), cdt, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) { (void)res; }
    else zip1<T, T>(aa, res, [&](T x) -> T {
      switch (op) {
        case UnOp::Neg: if constexpr (is_cplx<T>) return -x; else return static_cast<T>(-x);
        case UnOp::Pos: return x;
        case UnOp::Square: return static_cast<T>(x * x);
        case UnOp::Recip:
          if constexpr (std::is_integral_v<T>) return x == 0 ? 0 : static_cast<T>(1 / x);
          else return static_cast<T>(T(1) / x);
        case UnOp::Sign:
          if constexpr (is_cplx<T>) return std::abs(x) == 0.0 ? T(0) : x / std::abs(x);
          else if constexpr (std::is_unsigned_v<T>) return x > 0 ? T(1) : T(0);
          else return static_cast<T>((x > 0) - (x < 0));
      }
      return x;
    });
  });
  return cdt == out_dt ? res : res.astype(out_dt);
}

ndarray absolute_impl(const ndarray& a) {
  if (a.dtype().is_complex()) {
    DType out_dt = real_of(a.dtype());
    ndarray aa = a;  // read complex directly
    ndarray res(a.shape(), out_dt, Order::C);
    if (a.dtype() == kComplex64)
      zip1<std::complex<float>, float>(aa, res, [](std::complex<float> x) { return std::abs(x); });
    else
      zip1<std::complex<double>, double>(aa, res, [](std::complex<double> x) { return std::abs(x); });
    return res;
  }
  DType out_dt = a.dtype();
  DType cdt = proxy(out_dt);
  ndarray aa = a.astype(cdt);
  ndarray res(aa.shape(), cdt, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || is_cplx<T>) { (void)res; }
    else if constexpr (std::is_unsigned_v<T>) zip1<T, T>(aa, res, [](T x) { return x; });
    else zip1<T, T>(aa, res, [](T x) { return static_cast<T>(x < 0 ? -x : x); });
  });
  return cdt == out_dt ? res : res.astype(out_dt);
}

enum class FUn { Sqrt, Exp, Log, Sin, Cos, Tan, Floor, Ceil };
ndarray unary_float(const ndarray& a, FUn op) {
  if (a.dtype().is_complex() && (op == FUn::Floor || op == FUn::Ceil))
    throw type_error("floor/ceil not supported for complex dtype");
  DType out_dt = to_float(a.dtype());
  DType cdt = proxy(out_dt);  // half -> float32
  ndarray aa = a.astype(cdt);
  ndarray res(aa.shape(), cdt, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (!(std::is_floating_point_v<T> || is_cplx<T>)) { (void)res; }
    else zip1<T, T>(aa, res, [&](T x) -> T {
      using std::sqrt; using std::exp; using std::log; using std::sin; using std::cos; using std::tan;
      switch (op) {
        case FUn::Sqrt: return sqrt(x);
        case FUn::Exp: return exp(x);
        case FUn::Log: return log(x);
        case FUn::Sin: return sin(x);
        case FUn::Cos: return cos(x);
        case FUn::Tan: return tan(x);
        case FUn::Floor: if constexpr (is_cplx<T>) return x; else return std::floor(x);
        case FUn::Ceil: if constexpr (is_cplx<T>) return x; else return std::ceil(x);
      }
      return x;
    });
  });
  return cdt == out_dt ? res : res.astype(out_dt);
}

// ---- reductions ----
enum class RedOp { Sum, Prod, Min, Max, Mean, Any, All };

Shape reduced_shape(const Shape& s, std::optional<int64_t> axis, bool keepdims) {
  if (!axis) return keepdims ? Shape(s.size(), 1) : Shape{};
  int64_t ax = normalize_axis(*axis, static_cast<int64_t>(s.size()));
  Shape out;
  for (int64_t i = 0; i < static_cast<int64_t>(s.size()); ++i) {
    if (i == ax) { if (keepdims) out.push_back(1); }
    else out.push_back(s[i]);
  }
  return out;
}

// Arrange so the reduced axis is contiguous-last: returns (outer, L, contiguous data array).
ndarray axis_last_contiguous(const ndarray& a, std::optional<int64_t> axis, int64_t& outer, int64_t& L) {
  if (!axis) { outer = 1; L = a.size(); return a.ascontiguousarray(); }
  int64_t ax = normalize_axis(*axis, a.ndim());
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < a.ndim(); ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray t = a.transpose(perm).ascontiguousarray();
  L = a.shape()[ax];
  outer = a.size() == 0 ? 0 : a.size() / std::max<int64_t>(L, 1);
  return t;
}

template <class T>
T reduce_row_numeric(const T* p, int64_t n, RedOp op) {
  if constexpr (is_cplx<T>) {
    if (op == RedOp::Min || op == RedOp::Max)
      throw not_implemented_error("min/max reduction on complex not yet implemented");
  }
  T acc = (op == RedOp::Prod) ? T(1) : (op == RedOp::Sum || op == RedOp::Mean) ? T(0) : p[0];
  int64_t start = (op == RedOp::Min || op == RedOp::Max) ? 1 : 0;
  for (int64_t i = start; i < n; ++i) {
    T x = p[i];
    switch (op) {
      case RedOp::Sum: case RedOp::Mean: acc += x; break;
      case RedOp::Prod: acc *= x; break;
      case RedOp::Min: if constexpr (!is_cplx<T>) acc = std::min(acc, x); break;
      case RedOp::Max: if constexpr (!is_cplx<T>) acc = std::max(acc, x); break;
      default: break;
    }
  }
  if (op == RedOp::Mean) {
    if constexpr (is_cplx<T>) return acc / T(static_cast<typename T::value_type>(n));
    else return static_cast<T>(acc / static_cast<T>(n));
  }
  return acc;
}

DType reduce_compute_dtype(const ndarray& a, RedOp op, std::optional<DType> dt) {
  switch (op) {
    case RedOp::Sum: case RedOp::Prod:
      if (dt) return *dt;
      return is_int_kind(a.dtype()) ? default_int() : a.dtype();
    case RedOp::Mean:
      return is_int_kind(a.dtype()) ? kFloat64 : a.dtype();
    default:
      return a.dtype();
  }
}

ndarray reduce(const ndarray& a, RedOp op, std::optional<int64_t> axis, bool keepdims,
               std::optional<DType> dt) {
  if (op == RedOp::Any || op == RedOp::All) {
    int64_t outer = 0, L = 0;
    ndarray src = axis_last_contiguous(a.astype(kBool), axis, outer, L);
    ndarray flat(Shape{outer}, kBool, Order::C);
    const bool* p = src.size() ? src.typed_data<bool>() : nullptr;
    bool* o = outer ? flat.typed_data<bool>() : nullptr;
    for (int64_t r = 0; r < outer; ++r) {
      bool acc = (op == RedOp::All);
      for (int64_t i = 0; i < L; ++i) { bool v = p[r * L + i]; acc = (op == RedOp::All) ? (acc && v) : (acc || v); }
      o[r] = acc;
    }
    return flat.reshape(reduced_shape(a.shape(), axis, keepdims));
  }
  DType cdt = reduce_compute_dtype(a, op, dt);
  DType pdt = proxy(cdt);
  int64_t outer = 0, L = 0;
  ndarray src = axis_last_contiguous(a.astype(pdt), axis, outer, L);
  ndarray flat(Shape{outer}, pdt, Order::C);
  visit_dtype(pdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) { (void)flat; }
    else {
      const T* p = src.size() ? src.template typed_data<T>() : nullptr;
      T* o = outer ? flat.template typed_data<T>() : nullptr;
      for (int64_t r = 0; r < outer; ++r) o[r] = reduce_row_numeric<T>(p + r * L, L, op);
    }
  });
  ndarray shaped = flat.reshape(reduced_shape(a.shape(), axis, keepdims));
  return pdt == cdt ? shaped : shaped.astype(cdt);
}

}  // namespace

// ---- public API ----
ndarray add(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Add); }
ndarray subtract(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Sub); }
ndarray multiply(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Mul); }
ndarray divide(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Div); }
ndarray floor_divide(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::FloorDiv); }
ndarray mod(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Mod); }
ndarray power(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Power); }
ndarray minimum(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Min); }
ndarray maximum(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Max); }

ndarray equal(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Eq); }
ndarray not_equal(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Ne); }
ndarray less(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Lt); }
ndarray less_equal(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Le); }
ndarray greater(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Gt); }
ndarray greater_equal(const ndarray& a, const ndarray& b) { return compare(a, b, CmpOp::Ge); }

ndarray logical_and(const ndarray& a, const ndarray& b) { return logical(a, b, 0); }
ndarray logical_or(const ndarray& a, const ndarray& b) { return logical(a, b, 1); }
ndarray logical_xor(const ndarray& a, const ndarray& b) { return logical(a, b, 2); }
ndarray logical_not(const ndarray& a) {
  ndarray aa = a.astype(kBool);
  ndarray res(aa.shape(), kBool, Order::C);
  zip1<bool, bool>(aa, res, [](bool x) { return !x; });
  return res;
}
ndarray bitwise_and(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::And); }
ndarray bitwise_or(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Or); }
ndarray bitwise_xor(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Xor); }
ndarray invert(const ndarray& a) {
  if (a.dtype() == kBool) return logical_not(a);
  ndarray res(a.shape(), a.dtype(), Order::C);
  ndarray aa = a.ascontiguousarray();
  visit_dtype(a.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
      zip1<T, T>(aa, res, [](T x) { return static_cast<T>(~x); });
    else throw type_error("invert requires integer dtype");
  });
  return res;
}

ndarray negative(const ndarray& a) { return unary_same(a, UnOp::Neg); }
ndarray positive(const ndarray& a) { return unary_same(a, UnOp::Pos); }
ndarray absolute(const ndarray& a) { return absolute_impl(a); }
ndarray sign(const ndarray& a) { return unary_same(a, UnOp::Sign); }
ndarray square(const ndarray& a) { return unary_same(a, UnOp::Square); }
ndarray reciprocal(const ndarray& a) { return unary_same(a, UnOp::Recip); }
ndarray sqrt(const ndarray& a) { return unary_float(a, FUn::Sqrt); }
ndarray exp(const ndarray& a) { return unary_float(a, FUn::Exp); }
ndarray log(const ndarray& a) { return unary_float(a, FUn::Log); }
ndarray sin(const ndarray& a) { return unary_float(a, FUn::Sin); }
ndarray cos(const ndarray& a) { return unary_float(a, FUn::Cos); }
ndarray tan(const ndarray& a) { return unary_float(a, FUn::Tan); }
ndarray floor(const ndarray& a) { return unary_float(a, FUn::Floor); }
ndarray ceil(const ndarray& a) { return unary_float(a, FUn::Ceil); }

ndarray sum(const ndarray& a, std::optional<int64_t> axis, bool keepdims, std::optional<DType> dt) {
  return reduce(a, RedOp::Sum, axis, keepdims, dt);
}
ndarray prod(const ndarray& a, std::optional<int64_t> axis, bool keepdims, std::optional<DType> dt) {
  return reduce(a, RedOp::Prod, axis, keepdims, dt);
}
ndarray amin(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return reduce(a, RedOp::Min, axis, keepdims, std::nullopt);
}
ndarray amax(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return reduce(a, RedOp::Max, axis, keepdims, std::nullopt);
}
ndarray mean(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return reduce(a, RedOp::Mean, axis, keepdims, std::nullopt);
}
ndarray any(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return reduce(a, RedOp::Any, axis, keepdims, std::nullopt);
}
ndarray all(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return reduce(a, RedOp::All, axis, keepdims, std::nullopt);
}

}  // namespace numpp
