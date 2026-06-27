#include "numpp/umath/ufunc.hpp"

#include "numpp/backend/backend.hpp"
#include "numpp/backend/gpu_vtable.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/umath/errstate.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
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
enum class BinOp { Add, Sub, Mul, Div, FloorDiv, Mod, Power, Min, Max, And, Or, Xor,
                   Fmin, Fmax, LShift, RShift };

template <class T>
T int_floordiv(T x, T y) {
  if (y == 0) return 0;                       // numpy: int //0 -> 0 (with warning)
  if constexpr (std::is_signed_v<T>) {
    // MIN / -1 overflows (SIGFPE on x86); numpy wraps to MIN.
    if (x == std::numeric_limits<T>::min() && y == T(-1)) return x;
  }
  T q = x / y, r = x % y;
  if (r != 0 && ((r < 0) != (y < 0))) --q;    // round toward -inf
  return q;
}

template <class T>
T int_mod(T x, T y) {
  if (y == 0) return 0;                       // numpy: int %0 -> 0
  if constexpr (std::is_signed_v<T>) {
    if (x == std::numeric_limits<T>::min() && y == T(-1)) return 0;  // x % -1 == 0; avoid overflow
  }
  return static_cast<T>(x - int_floordiv<T>(x, y) * y);
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
      case BinOp::Mod: return int_mod<T>(x, y);
      case BinOp::Power:
        if constexpr (std::is_signed_v<T>) {
          if (y < 0) throw value_error("Integers to negative integer powers are not allowed.");
        }
        return static_cast<T>(std::pow(static_cast<double>(x), static_cast<double>(y)));
      case BinOp::Min: case BinOp::Fmin: return std::min(x, y);
      case BinOp::Max: case BinOp::Fmax: return std::max(x, y);
      case BinOp::And: return static_cast<T>(x & y);
      case BinOp::Or: return static_cast<T>(x | y);
      case BinOp::Xor: return static_cast<T>(x ^ y);
      case BinOp::LShift: return static_cast<T>(x << y);
      case BinOp::RShift: return static_cast<T>(x >> y);
      default: throw type_error("unsupported integer binary op");
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
      // minimum/maximum propagate NaN; fmin/fmax ignore it (NumPy semantics).
      case BinOp::Min: return std::isnan(x) ? x : std::isnan(y) ? y : std::min(x, y);
      case BinOp::Max: return std::isnan(x) ? x : std::isnan(y) ? y : std::max(x, y);
      case BinOp::Fmin: return std::fmin(x, y);
      case BinOp::Fmax: return std::fmax(x, y);
      default: throw type_error("bitwise/shift operation requires integer dtype");
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
    case BinOp::Min: case BinOp::Max: case BinOp::Fmin: case BinOp::Fmax:
      if (p.is_complex()) throw not_implemented_error("min/max on complex not yet implemented");
      return p;
    case BinOp::And: case BinOp::Or: case BinOp::Xor:
      if (p.is_floating() || p.is_complex()) throw type_error("bitwise op requires integer/bool dtype");
      return p;
    case BinOp::LShift: case BinOp::RShift:
      if (p.is_floating() || p.is_complex()) throw type_error("shift requires integer dtype");
      return p == kBool ? kInt8 : p;  // numpy promotes bool shifts to int8
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
  const bool track_fp = cdt.is_floating() || cdt.is_complex();
  detail::fp_guard fpg;
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    // half and bool are proxied away before compute; guard so the switch compiles.
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) { (void)res; }
    else zip2<T, T>(aa, bb, res, [&](T x, T y) { return bin_scalar<T>(op, x, y); });
  });
  if (track_fp) fpg.check();  // apply errstate policy (may throw on "raise")
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
          else if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(x)) return x;  // numpy: sign(nan) == nan
            return static_cast<T>((x > 0) - (x < 0));
          } else return static_cast<T>((x > 0) - (x < 0));
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

enum class FUn { Sqrt, Cbrt, Exp, Expm1, Log, Log2, Log10, Log1p, Sin, Cos, Tan,
                 Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
                 Deg2rad, Rad2deg, Floor, Ceil, Trunc, Rint };

template <class T>  // T is float or double
T funary_real(FUn op, T x) {
  constexpr T pi = static_cast<T>(3.14159265358979323846);
  switch (op) {
    case FUn::Sqrt: return std::sqrt(x);   case FUn::Cbrt: return std::cbrt(x);
    case FUn::Exp: return std::exp(x);     case FUn::Expm1: return std::expm1(x);
    case FUn::Log: return std::log(x);     case FUn::Log2: return std::log2(x);
    case FUn::Log10: return std::log10(x); case FUn::Log1p: return std::log1p(x);
    case FUn::Sin: return std::sin(x);     case FUn::Cos: return std::cos(x);
    case FUn::Tan: return std::tan(x);     case FUn::Asin: return std::asin(x);
    case FUn::Acos: return std::acos(x);   case FUn::Atan: return std::atan(x);
    case FUn::Sinh: return std::sinh(x);   case FUn::Cosh: return std::cosh(x);
    case FUn::Tanh: return std::tanh(x);   case FUn::Asinh: return std::asinh(x);
    case FUn::Acosh: return std::acosh(x); case FUn::Atanh: return std::atanh(x);
    case FUn::Deg2rad: return x * pi / static_cast<T>(180);
    case FUn::Rad2deg: return x * static_cast<T>(180) / pi;
    case FUn::Floor: return std::floor(x); case FUn::Ceil: return std::ceil(x);
    case FUn::Trunc: return std::trunc(x); case FUn::Rint: return std::rint(x);
  }
  return x;
}

template <class C>  // C is std::complex<...>
C funary_cplx(FUn op, C x) {
  switch (op) {
    case FUn::Sqrt: return std::sqrt(x);  case FUn::Exp: return std::exp(x);
    case FUn::Log: return std::log(x);    case FUn::Log10: return std::log10(x);
    case FUn::Log2: return std::log(x) / std::log(C(2));
    case FUn::Log1p: return std::log(C(1) + x);
    case FUn::Sin: return std::sin(x);    case FUn::Cos: return std::cos(x);
    case FUn::Tan: return std::tan(x);    case FUn::Asin: return std::asin(x);
    case FUn::Acos: return std::acos(x);  case FUn::Atan: return std::atan(x);
    case FUn::Sinh: return std::sinh(x);  case FUn::Cosh: return std::cosh(x);
    case FUn::Tanh: return std::tanh(x);  case FUn::Asinh: return std::asinh(x);
    case FUn::Acosh: return std::acosh(x);case FUn::Atanh: return std::atanh(x);
    default: throw type_error("ufunc not supported for complex dtype");
  }
}

ndarray unary_float(const ndarray& a, FUn op) {
  DType out_dt = to_float(a.dtype());
  DType cdt = proxy(out_dt);  // half -> float32
  ndarray aa = a.astype(cdt);
  ndarray res(aa.shape(), cdt, Order::C);
  detail::fp_guard fpg;
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_floating_point_v<T>) zip1<T, T>(aa, res, [&](T x) { return funary_real<T>(op, x); });
    else if constexpr (is_cplx<T>) zip1<T, T>(aa, res, [&](T x) { return funary_cplx<T>(op, x); });
    else { (void)res; }
  });
  fpg.check();  // apply errstate policy (may throw on "raise")
  return cdt == out_dt ? res : res.astype(out_dt);
}

// binary float ufuncs (real only): arctan2, hypot, copysign
enum class BinF { Atan2, Hypot, Copysign };
ndarray binary_float(const ndarray& a, const ndarray& b, BinF op) {
  DType p = result_type(a.dtype(), b.dtype());
  if (p.is_complex()) throw type_error("arctan2/hypot/copysign not supported for complex");
  DType out_dt = to_float(p);
  DType cdt = proxy(out_dt);
  Shape bs = broadcast_shapes(a.shape(), b.shape());
  ndarray aa = a.astype(cdt).broadcast_to(bs);
  ndarray bb = b.astype(cdt).broadcast_to(bs);
  ndarray res(bs, cdt, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_floating_point_v<T>)
      zip2<T, T>(aa, bb, res, [&](T x, T y) {
        switch (op) { case BinF::Atan2: return std::atan2(x, y);
                      case BinF::Hypot: return std::hypot(x, y);
                      case BinF::Copysign: return std::copysign(x, y); }
        return T{};
      });
    else { (void)res; }
  });
  return cdt == out_dt ? res : res.astype(out_dt);
}

// predicates (bool result): isnan, isinf, isfinite, signbit
enum class Pred { Isnan, Isinf, Isfinite, Signbit };
ndarray unary_pred(const ndarray& a, Pred op) {
  if (a.dtype().is_complex() && op == Pred::Signbit)
    throw type_error("signbit not supported for complex");
  DType cdt = proxy(to_float(a.dtype()));
  ndarray aa = a.astype(cdt);
  ndarray res(aa.shape(), kBool, Order::C);
  visit_dtype(cdt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_floating_point_v<T>)
      zip1<T, bool>(aa, res, [&](T x) {
        switch (op) { case Pred::Isnan: return std::isnan(x); case Pred::Isinf: return std::isinf(x);
                      case Pred::Isfinite: return std::isfinite(x); case Pred::Signbit: return std::signbit(x); }
        return false;
      });
    else if constexpr (is_cplx<T>)
      zip1<T, bool>(aa, res, [&](T x) {
        switch (op) {
          case Pred::Isnan: return std::isnan(x.real()) || std::isnan(x.imag());
          case Pred::Isinf: return std::isinf(x.real()) || std::isinf(x.imag());
          case Pred::Isfinite: return std::isfinite(x.real()) && std::isfinite(x.imag());
          default: return false;
        }
      });
    else { (void)res; }
  });
  return res;
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

// ---- optional GPU dispatch (weak vtable; CPU kernel is always the fallback) ----
bool gpu_dtype_ok(DType d) { return d == kFloat32 || d == kFloat64 || d == kComplex64 || d == kComplex128; }
int64_t gpu_min() {
  if (const char* e = std::getenv("NUMPP_GPU_MIN")) { char* end = nullptr; long long v = std::strtoll(e, &end, 10); if (end != e && v >= 0) return v; }
  return 1 << 16;
}
ndarray gpu_or_binary(const ndarray& a, const ndarray& b, BinOp op, int gop, const std::function<ndarray()>& cpu) {
  const GpuVTable* vt = gpu_vtable();
  DType out_dt = arith_out(op, a.dtype(), b.dtype());
  if (vt && gpu_dtype_ok(out_dt) && a.shape() == b.shape() && a.size() >= gpu_min()) {
    ndarray ac = a.astype(out_dt).ascontiguousarray(), bc = b.astype(out_dt).ascontiguousarray();
    ndarray out(ac.shape(), out_dt, Order::C);
    if (vt->elementwise_binary(gop, out_dt.id(), out.size(), ac.bytes(), bc.bytes(), out.bytes())) { set_last_backend(Backend::Device); return out; }
  }
  set_last_backend(Backend::Cpu);
  return cpu();
}
ndarray gpu_or_unary(const ndarray& a, int gop, DType out_dt, const std::function<ndarray()>& cpu) {
  const GpuVTable* vt = gpu_vtable();
  if (vt && gpu_dtype_ok(out_dt) && a.size() >= gpu_min()) {
    ndarray ac = a.astype(out_dt).ascontiguousarray();
    ndarray out(ac.shape(), out_dt, Order::C);
    if (vt->elementwise_unary(gop, out_dt.id(), out.size(), ac.bytes(), out.bytes())) { set_last_backend(Backend::Device); return out; }
  }
  set_last_backend(Backend::Cpu);
  return cpu();
}
}  // namespace

// ---- public API ----
ndarray add(const ndarray& a, const ndarray& b) { return gpu_or_binary(a, b, BinOp::Add, kGAdd, [&] { return binary(a, b, BinOp::Add); }); }
ndarray subtract(const ndarray& a, const ndarray& b) { return gpu_or_binary(a, b, BinOp::Sub, kGSub, [&] { return binary(a, b, BinOp::Sub); }); }
ndarray multiply(const ndarray& a, const ndarray& b) { return gpu_or_binary(a, b, BinOp::Mul, kGMul, [&] { return binary(a, b, BinOp::Mul); }); }
ndarray divide(const ndarray& a, const ndarray& b) { return gpu_or_binary(a, b, BinOp::Div, kGDiv, [&] { return binary(a, b, BinOp::Div); }); }
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

ndarray negative(const ndarray& a) { return gpu_or_unary(a, kGNeg, a.dtype(), [&] { return unary_same(a, UnOp::Neg); }); }
ndarray positive(const ndarray& a) { return unary_same(a, UnOp::Pos); }
ndarray absolute(const ndarray& a) { return absolute_impl(a); }
ndarray sign(const ndarray& a) { return unary_same(a, UnOp::Sign); }
ndarray square(const ndarray& a) { return unary_same(a, UnOp::Square); }
ndarray reciprocal(const ndarray& a) { return unary_same(a, UnOp::Recip); }
ndarray sqrt(const ndarray& a) { return gpu_or_unary(a, kGSqrt, to_float(a.dtype()), [&] { return unary_float(a, FUn::Sqrt); }); }
ndarray exp(const ndarray& a) { return gpu_or_unary(a, kGExp, to_float(a.dtype()), [&] { return unary_float(a, FUn::Exp); }); }
ndarray log(const ndarray& a) { return unary_float(a, FUn::Log); }
ndarray sin(const ndarray& a) { return unary_float(a, FUn::Sin); }
ndarray cos(const ndarray& a) { return unary_float(a, FUn::Cos); }
ndarray tan(const ndarray& a) { return unary_float(a, FUn::Tan); }
ndarray floor(const ndarray& a) { return unary_float(a, FUn::Floor); }
ndarray ceil(const ndarray& a) { return unary_float(a, FUn::Ceil); }

namespace {
ndarray gpu_or_reduce(const ndarray& a, int gop, std::optional<int64_t> axis, bool keepdims,
                      std::optional<DType> dt, const std::function<ndarray()>& cpu) {
  const GpuVTable* vt = gpu_vtable();
  if (!axis && !keepdims && !dt && vt && gpu_dtype_ok(a.dtype()) && a.size() >= gpu_min()) {
    ndarray ac = a.ascontiguousarray();
    ndarray out(Shape{}, a.dtype(), Order::C);
    if (vt->reduce(gop, a.dtype().id(), ac.size(), ac.bytes(), out.bytes())) { set_last_backend(Backend::Device); return out; }
  }
  set_last_backend(Backend::Cpu);
  return cpu();
}
}  // namespace
ndarray sum(const ndarray& a, std::optional<int64_t> axis, bool keepdims, std::optional<DType> dt) {
  return gpu_or_reduce(a, kGSum, axis, keepdims, dt, [&] { return reduce(a, RedOp::Sum, axis, keepdims, dt); });
}
ndarray prod(const ndarray& a, std::optional<int64_t> axis, bool keepdims, std::optional<DType> dt) {
  return gpu_or_reduce(a, kGProd, axis, keepdims, dt, [&] { return reduce(a, RedOp::Prod, axis, keepdims, dt); });
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

// NaN-ignoring reductions (compose existing ufuncs; NaN treated as identity).
ndarray nansum(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  return sum(where(isnan(a), zeros_like(a), a), axis, keepdims);
}
ndarray nanmean(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  ndarray isn = isnan(a);
  ndarray s = sum(where(isn, zeros_like(a), a), axis, keepdims).astype(kFloat64);
  ndarray cnt = sum(logical_not(isn), axis, keepdims).astype(kFloat64);
  return divide(s, cnt);
}
namespace {
// All-NaN slices reduce to +/-inf with the fill trick; numpy returns NaN there.
ndarray nan_extreme_fix(const ndarray& r, const ndarray& isn, std::optional<int64_t> axis, bool keepdims) {
  ndarray cnt = sum(logical_not(isn), axis, keepdims);  // count of non-NaN per slice
  ndarray nanfill = full_like(r, std::numeric_limits<double>::quiet_NaN());
  return where(equal(cnt, full_like(cnt, 0)), nanfill, r);
}
}  // namespace

ndarray nanmin(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  if (!a.dtype().is_floating()) return amin(a, axis, keepdims);
  double inf = std::numeric_limits<double>::infinity();
  ndarray isn = isnan(a);
  ndarray r = amin(where(isn, full_like(a, inf), a), axis, keepdims);
  return nan_extreme_fix(r, isn, axis, keepdims);
}
ndarray nanmax(const ndarray& a, std::optional<int64_t> axis, bool keepdims) {
  if (!a.dtype().is_floating()) return amax(a, axis, keepdims);
  double inf = std::numeric_limits<double>::infinity();
  ndarray isn = isnan(a);
  ndarray r = amax(where(isn, full_like(a, -inf), a), axis, keepdims);
  return nan_extreme_fix(r, isn, axis, keepdims);
}
ndarray nanvar(const ndarray& a, std::optional<int64_t> axis, bool keepdims, int64_t ddof) {
  ndarray isn = isnan(a);
  ndarray cnt = sum(logical_not(isn), axis, /*keepdims=*/true).astype(kFloat64);
  ndarray af = a.astype(kFloat64);
  ndarray mean = divide(sum(where(isn, zeros_like(af), af), axis, true), cnt);
  ndarray dev = where(isn, zeros_like(af), square(subtract(af, mean)));
  ndarray ssum = sum(dev, axis, keepdims);
  ndarray denom = subtract(keepdims ? cnt : sum(logical_not(isn), axis, false).astype(kFloat64),
                           scalar_like(static_cast<double>(ddof), kFloat64, true));
  return divide(ssum, denom);
}
ndarray nanstd(const ndarray& a, std::optional<int64_t> axis, bool keepdims, int64_t ddof) {
  return sqrt(nanvar(a, axis, keepdims, ddof));
}

ndarray fmin(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Fmin); }
ndarray fmax(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::Fmax); }
ndarray left_shift(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::LShift); }
ndarray right_shift(const ndarray& a, const ndarray& b) { return binary(a, b, BinOp::RShift); }
ndarray arctan2(const ndarray& a, const ndarray& b) { return binary_float(a, b, BinF::Atan2); }
ndarray hypot(const ndarray& a, const ndarray& b) { return binary_float(a, b, BinF::Hypot); }
ndarray copysign(const ndarray& a, const ndarray& b) { return binary_float(a, b, BinF::Copysign); }

ndarray cbrt(const ndarray& a) { return unary_float(a, FUn::Cbrt); }
ndarray expm1(const ndarray& a) { return unary_float(a, FUn::Expm1); }
ndarray log2(const ndarray& a) { return unary_float(a, FUn::Log2); }
ndarray log10(const ndarray& a) { return unary_float(a, FUn::Log10); }
ndarray log1p(const ndarray& a) { return unary_float(a, FUn::Log1p); }
ndarray arcsin(const ndarray& a) { return unary_float(a, FUn::Asin); }
ndarray arccos(const ndarray& a) { return unary_float(a, FUn::Acos); }
ndarray arctan(const ndarray& a) { return unary_float(a, FUn::Atan); }
ndarray sinh(const ndarray& a) { return unary_float(a, FUn::Sinh); }
ndarray cosh(const ndarray& a) { return unary_float(a, FUn::Cosh); }
ndarray tanh(const ndarray& a) { return unary_float(a, FUn::Tanh); }
ndarray arcsinh(const ndarray& a) { return unary_float(a, FUn::Asinh); }
ndarray arccosh(const ndarray& a) { return unary_float(a, FUn::Acosh); }
ndarray arctanh(const ndarray& a) { return unary_float(a, FUn::Atanh); }
ndarray deg2rad(const ndarray& a) { return unary_float(a, FUn::Deg2rad); }
ndarray rad2deg(const ndarray& a) { return unary_float(a, FUn::Rad2deg); }
ndarray trunc(const ndarray& a) { return unary_float(a, FUn::Trunc); }
ndarray rint(const ndarray& a) { return unary_float(a, FUn::Rint); }

ndarray isnan(const ndarray& a) { return unary_pred(a, Pred::Isnan); }
ndarray isinf(const ndarray& a) { return unary_pred(a, Pred::Isinf); }
ndarray isfinite(const ndarray& a) { return unary_pred(a, Pred::Isfinite); }
ndarray signbit(const ndarray& a) { return unary_pred(a, Pred::Signbit); }

ndarray clip(const ndarray& a, const ndarray& lo, const ndarray& hi) {
  return minimum(maximum(a, lo), hi);
}

ndarray conj(const ndarray& a) {
  if (!a.dtype().is_complex()) return a.copy();
  ndarray res(a.shape(), a.dtype(), Order::C);
  if (a.dtype() == kComplex64)
    zip1<std::complex<float>, std::complex<float>>(a, res, [](std::complex<float> x) { return std::conj(x); });
  else
    zip1<std::complex<double>, std::complex<double>>(a, res, [](std::complex<double> x) { return std::conj(x); });
  return res;
}
ndarray conjugate(const ndarray& a) { return conj(a); }

ndarray real(const ndarray& a) {
  if (!a.dtype().is_complex()) return a.copy();
  DType out_dt = a.dtype() == kComplex64 ? kFloat32 : kFloat64;
  ndarray res(a.shape(), out_dt, Order::C);
  if (a.dtype() == kComplex64)
    zip1<std::complex<float>, float>(a, res, [](std::complex<float> x) { return x.real(); });
  else
    zip1<std::complex<double>, double>(a, res, [](std::complex<double> x) { return x.real(); });
  return res;
}
ndarray imag(const ndarray& a) {
  if (!a.dtype().is_complex()) return zeros_like(a);
  DType out_dt = a.dtype() == kComplex64 ? kFloat32 : kFloat64;
  ndarray res(a.shape(), out_dt, Order::C);
  if (a.dtype() == kComplex64)
    zip1<std::complex<float>, float>(a, res, [](std::complex<float> x) { return x.imag(); });
  else
    zip1<std::complex<double>, double>(a, res, [](std::complex<double> x) { return x.imag(); });
  return res;
}
ndarray angle(const ndarray& a) {
  if (a.dtype().is_complex()) {
    DType out_dt = a.dtype() == kComplex64 ? kFloat32 : kFloat64;
    ndarray res(a.shape(), out_dt, Order::C);
    if (a.dtype() == kComplex64)
      zip1<std::complex<float>, float>(a, res, [](std::complex<float> x) { return std::arg(x); });
    else
      zip1<std::complex<double>, double>(a, res, [](std::complex<double> x) { return std::arg(x); });
    return res;
  }
  return arctan2(zeros_like(a), a);  // angle of real x = atan2(0, x): 0 if x>=0 else pi
}

void copyto(ndarray& dst, const ndarray& src, const ndarray* where) {
  if (!dst.writeable()) throw value_error("copyto destination is read-only");
  ndarray s = src.astype(dst.dtype()).broadcast_to(dst.shape());
  const bool hasMask = where != nullptr;
  ndarray mask;
  if (hasMask) mask = where->astype(kBool).broadcast_to(dst.shape());
  const int64_t nd = dst.ndim(), total = dst.size();
  if (total == 0) return;
  visit_dtype(dst.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    std::vector<int64_t> idx(nd, 0);
    while (true) {
      int64_t doff = dst.offset(), soff = s.offset(), moff = hasMask ? mask.offset() : 0;
      for (int64_t i = 0; i < nd; ++i) {
        doff += idx[i] * dst.strides()[i];
        soff += idx[i] * s.strides()[i];
        if (hasMask) moff += idx[i] * mask.strides()[i];
      }
      bool take = true;
      if (hasMask) { std::memcpy(&take, mask.buffer()->data() + moff, 1); }
      if (take) std::memcpy(dst.buffer()->data() + doff, s.buffer()->data() + soff, sizeof(T));
      int64_t ax = nd - 1;
      for (; ax >= 0; --ax) { if (++idx[ax] < dst.shape()[ax]) break; idx[ax] = 0; }
      if (ax < 0) break;
    }
  });
}

ndarray add(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where) {
  copyto(out, add(a, b), where); return out;
}
ndarray subtract(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where) {
  copyto(out, subtract(a, b), where); return out;
}
ndarray multiply(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where) {
  copyto(out, multiply(a, b), where); return out;
}
ndarray divide(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where) {
  copyto(out, divide(a, b), where); return out;
}

ndarray scalar_like(double value, DType like, bool is_float) {
  DType sdt;
  if (is_float) sdt = (like.is_floating() || like.is_complex()) ? like : kFloat64;
  else sdt = (like == kBool) ? default_int() : like;
  ndarray tmp(Shape{}, kFloat64, Order::C);
  tmp.set_item<double>({}, value);
  return tmp.astype(sdt, Casting::Unsafe);
}

ndarray var(const ndarray& a, std::optional<int64_t> axis, bool keepdims, int64_t ddof) {
  int64_t n = axis ? a.shape()[normalize_axis(*axis, a.ndim())] : a.size();
  ndarray m = mean(a, axis, /*keepdims=*/true);
  ndarray sq = square(absolute(subtract(a, m)));         // real, nonneg; complex-safe
  ndarray s = sum(sq, axis, keepdims);
  ndarray denom = scalar_like(static_cast<double>(n - ddof), s.dtype(), true);
  return divide(s, denom);
}
ndarray std(const ndarray& a, std::optional<int64_t> axis, bool keepdims, int64_t ddof) {
  return sqrt(var(a, axis, keepdims, ddof));
}

ndarray where(const ndarray& cond, const ndarray& a, const ndarray& b) {
  DType odt = result_type(a.dtype(), b.dtype());
  Shape bs = broadcast_shapes(broadcast_shapes(cond.shape(), a.shape()), b.shape());
  ndarray cc = cond.astype(kBool).broadcast_to(bs);
  ndarray aa = a.astype(odt).broadcast_to(bs);
  ndarray bb = b.astype(odt).broadcast_to(bs);
  ndarray res(bs, odt, Order::C);
  const int64_t nd = static_cast<int64_t>(bs.size()), total = shape_size(bs);
  if (total == 0) return res;
  visit_dtype(odt.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* out = res.template typed_data<T>();
    std::vector<int64_t> idx(nd, 0);
    int64_t lin = 0;
    while (true) {
      int64_t co = cc.offset(), ao = aa.offset(), bo = bb.offset();
      for (int64_t i = 0; i < nd; ++i) { co += idx[i] * cc.strides()[i]; ao += idx[i] * aa.strides()[i]; bo += idx[i] * bb.strides()[i]; }
      bool c; T av, bv;
      std::memcpy(&c, cc.buffer()->data() + co, 1);
      std::memcpy(&av, aa.buffer()->data() + ao, sizeof(T));
      std::memcpy(&bv, bb.buffer()->data() + bo, sizeof(T));
      out[lin++] = c ? av : bv;
      int64_t ax = nd - 1;
      for (; ax >= 0; --ax) { if (++idx[ax] < bs[ax]) break; idx[ax] = 0; }
      if (ax < 0) break;
    }
  });
  return res;
}

std::vector<ndarray> nonzero(const ndarray& a) {
  ndarray mask = a.astype(kBool).ascontiguousarray();
  const int64_t nd = a.ndim(), total = a.size();
  const bool* m = total ? mask.typed_data<bool>() : nullptr;
  std::vector<std::vector<int64_t>> coords(nd);
  std::vector<int64_t> idx(nd, 0);
  for (int64_t lin = 0; lin < total; ++lin) {
    if (m[lin]) for (int64_t d = 0; d < nd; ++d) coords[d].push_back(idx[d]);
    int64_t ax = nd - 1;
    for (; ax >= 0; --ax) { if (++idx[ax] < a.shape()[ax]) break; idx[ax] = 0; }
  }
  std::vector<ndarray> out;
  for (int64_t d = 0; d < nd; ++d) {
    ndarray col(Shape{static_cast<int64_t>(coords[d].size())}, kInt64, Order::C);
    for (size_t i = 0; i < coords[d].size(); ++i) col.set_item<int64_t>({static_cast<int64_t>(i)}, coords[d][i]);
    out.push_back(col);
  }
  return out;
}

}  // namespace numpp
