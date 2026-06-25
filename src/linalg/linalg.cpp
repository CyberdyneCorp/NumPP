#include "numpp/linalg/linalg.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "numpp/backend/backend.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/umath/ufunc.hpp"

namespace numpp {
namespace {

// Output dtype per numpy.linalg: float32 stays float32, complex stays complex,
// everything else -> float64. Computation is always done in double precision.
struct Kind { DType compute; DType out; bool cmplx; };
Kind la_kind(DType d) {
  bool c = d.is_complex();
  DType out = c ? (d == kComplex64 ? kComplex64 : kComplex128)
                : (d == kFloat32 ? kFloat32 : kFloat64);
  return {c ? kComplex128 : kFloat64, out, c};
}

template <class T> T conj_(T x) { if constexpr (std::is_same_v<T, std::complex<double>>) return std::conj(x); else return x; }

void require_square(const ndarray& a) {
  if (a.ndim() != 2 || a.shape()[0] != a.shape()[1])
    throw linalg_error("Last 2 dimensions of the array must be square");
}

template <class T>
std::vector<T> to_vec(const ndarray& a, DType compute) {
  ndarray c = a.astype(compute).ascontiguousarray();
  const T* p = c.size() ? c.template typed_data<T>() : nullptr;
  return std::vector<T>(p, p + c.size());
}
template <class T>
ndarray from_vec(const std::vector<T>& v, Shape shape, DType compute, DType out) {
  ndarray r(shape, compute, Order::C);
  if (!v.empty()) std::memcpy(r.bytes(), v.data(), v.size() * sizeof(T));
  return r.astype(out);
}

// LU decomposition with partial pivoting (row-major n*n, in place). false=singular.
template <class T>
bool lu_decompose(std::vector<T>& A, int n, std::vector<int>& piv, double& permsign) {
  piv.resize(n);
  for (int i = 0; i < n; ++i) piv[i] = i;
  permsign = 1.0;
  for (int k = 0; k < n; ++k) {
    int pr = k; double best = std::abs(A[k * n + k]);
    for (int i = k + 1; i < n; ++i) { double v = std::abs(A[i * n + k]); if (v > best) { best = v; pr = i; } }
    if (best == 0.0) return false;
    if (pr != k) {
      for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[pr * n + j]);
      std::swap(piv[k], piv[pr]);
      permsign = -permsign;
    }
    const T akk = A[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      T f = A[i * n + k] / akk;
      A[i * n + k] = f;
      for (int j = k + 1; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
    }
  }
  return true;
}

// Solve using a computed LU factorization for an n*m right-hand side (row-major).
template <class T>
void lu_solve(const std::vector<T>& LU, int n, const std::vector<int>& piv,
              std::vector<T>& B, int m) {
  std::vector<T> X(static_cast<size_t>(n) * m);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) X[i * m + j] = B[piv[i] * m + j];
  for (int i = 0; i < n; ++i)                          // forward (unit lower)
    for (int k = 0; k < i; ++k)
      for (int j = 0; j < m; ++j) X[i * m + j] -= LU[i * n + k] * X[k * m + j];
  for (int i = n - 1; i >= 0; --i) {                   // back (upper)
    for (int k = i + 1; k < n; ++k)
      for (int j = 0; j < m; ++j) X[i * m + j] -= LU[i * n + k] * X[k * m + j];
    T d = LU[i * n + i];
    for (int j = 0; j < m; ++j) X[i * m + j] /= d;
  }
  B.swap(X);
}

template <class T>
ndarray solve_impl(const ndarray& a, const ndarray& b, Kind k) {
  const int n = static_cast<int>(a.shape()[0]);
  const bool bvec = b.ndim() == 1;
  const int m = bvec ? 1 : static_cast<int>(b.shape()[1]);
  std::vector<T> A = to_vec<T>(a, k.compute);
  std::vector<T> B = to_vec<T>(b, k.compute);
  std::vector<int> piv; double sgn;
  if (!lu_decompose<T>(A, n, piv, sgn)) throw linalg_error("Singular matrix");
  lu_solve<T>(A, n, piv, B, m);
  return from_vec<T>(B, b.shape(), k.compute, k.out);
}

template <class T>
ndarray inv_impl(const ndarray& a, Kind k) {
  const int n = static_cast<int>(a.shape()[0]);
  std::vector<T> A = to_vec<T>(a, k.compute);
  std::vector<int> piv; double sgn;
  if (!lu_decompose<T>(A, n, piv, sgn)) throw linalg_error("Singular matrix");
  std::vector<T> I(static_cast<size_t>(n) * n, T(0));
  for (int i = 0; i < n; ++i) I[i * n + i] = T(1);
  lu_solve<T>(A, n, piv, I, n);
  return from_vec<T>(I, a.shape(), k.compute, k.out);
}

template <class T>
ndarray cholesky_impl(const ndarray& a, Kind k) {
  const int n = static_cast<int>(a.shape()[0]);
  std::vector<T> A = to_vec<T>(a, k.compute);
  std::vector<T> L(static_cast<size_t>(n) * n, T(0));
  for (int j = 0; j < n; ++j) {
    T sum = A[j * n + j];
    for (int p = 0; p < j; ++p) sum -= L[j * n + p] * conj_(L[j * n + p]);
    double diag = std::real(std::complex<double>(sum));
    if (diag <= 0.0) throw linalg_error("Matrix is not positive definite");
    T ljj = T(std::sqrt(diag));
    L[j * n + j] = ljj;
    for (int i = j + 1; i < n; ++i) {
      T s = A[i * n + j];
      for (int p = 0; p < j; ++p) s -= L[i * n + p] * conj_(L[j * n + p]);
      L[i * n + j] = s / ljj;
    }
  }
  return from_vec<T>(L, a.shape(), k.compute, k.out);
}

// Dispatch helper: pick double vs complex<double> path from a dtype.
template <class F>
ndarray dispatch(DType d, F&& f) {
  Kind k = la_kind(d);
  if (k.cmplx) return f(std::complex<double>{}, k);
  return f(double{}, k);
}

}  // namespace

ndarray dot(const ndarray& a, const ndarray& b) {
  if (a.ndim() == 1 && b.ndim() == 1) return sum(multiply(a, b));
  if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b);
  if (a.ndim() == 2 && b.ndim() == 1)
    return matmul(a, b.reshape({b.size(), 1})).reshape({a.shape()[0]});
  if (a.ndim() == 1 && b.ndim() == 2)
    return matmul(a.reshape({1, a.size()}), b).reshape({b.shape()[1]});
  throw not_implemented_error("dot for ndim>2 not yet implemented");
}
ndarray vdot(const ndarray& a, const ndarray& b) {
  return sum(multiply(conj(a.ravel()), b.ravel()));
}
ndarray inner(const ndarray& a, const ndarray& b) {
  if (a.ndim() == 1 && b.ndim() == 1) return sum(multiply(a, b));
  if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b.transpose());
  throw not_implemented_error("inner for these shapes not yet implemented");
}
ndarray outer(const ndarray& a, const ndarray& b) {
  ndarray af = a.ravel(), bf = b.ravel();
  return multiply(af.reshape({af.size(), 1}), bf.reshape({1, bf.size()}));
}
ndarray trace(const ndarray& a, int64_t offset) {
  if (a.ndim() != 2) throw not_implemented_error("trace requires a 2-D array");
  const int64_t r = a.shape()[0], c = a.shape()[1];
  DType dt = a.dtype();
  if (dt.kind() == 'i' || dt.kind() == 'u' || dt == kBool) dt = default_int();
  ndarray acc = zeros({}, dt);
  for (int64_t i = 0; i < r; ++i) {
    int64_t j = i + offset;
    if (j >= 0 && j < c) acc = add(acc, a.index({IndexItem{i}, IndexItem{j}}).astype(dt));
  }
  return acc;
}
ndarray kron(const ndarray& a, const ndarray& b) {
  if (a.ndim() == 1 && b.ndim() == 1) {
    ndarray r = outer(a, b);  // (n,m)
    return r.reshape({a.size() * b.size()});
  }
  if (a.ndim() == 2 && b.ndim() == 2) {
    const int64_t am = a.shape()[0], an = a.shape()[1], bm = b.shape()[0], bn = b.shape()[1];
    DType dt = result_type(a.dtype(), b.dtype());
    ndarray out = zeros({am * bm, an * bn}, dt);
    ndarray aa = a.astype(dt), bb = b.astype(dt);
    for (int64_t i = 0; i < am; ++i)
      for (int64_t j = 0; j < an; ++j) {
        ndarray scaled = multiply(bb, aa.index({IndexItem{i}, IndexItem{j}}));  // bm x bn
        ndarray block = out.index({IndexItem{Slice{i * bm, (i + 1) * bm, 1}},
                                   IndexItem{Slice{j * bn, (j + 1) * bn, 1}}});
        copyto(block, scaled);
      }
    return out;
  }
  throw not_implemented_error("kron for ndim>2 not yet implemented");
}

namespace linalg {

ndarray solve(const ndarray& a, const ndarray& b) {
  require_square(a);
  Kind k = la_kind(result_type(a.dtype(), b.dtype()));
  if (k.cmplx) return solve_impl<std::complex<double>>(a, b, k);
  return solve_impl<double>(a, b, k);
}
ndarray inv(const ndarray& a) {
  require_square(a);
  return dispatch(a.dtype(), [&](auto tag, Kind k) {
    using T = decltype(tag); return inv_impl<T>(a, k);
  });
}
ndarray det(const ndarray& a) {
  require_square(a);
  return dispatch(a.dtype(), [&](auto tag, Kind k) -> ndarray {
    using T = decltype(tag);
    const int n = static_cast<int>(a.shape()[0]);
    std::vector<T> A = to_vec<T>(a, k.compute);
    std::vector<int> piv; double sgn;
    T d;
    if (!lu_decompose<T>(A, n, piv, sgn)) d = T(0);
    else { d = T(sgn); for (int i = 0; i < n; ++i) d *= A[i * n + i]; }
    std::vector<T> v{d};
    return from_vec<T>(v, {}, k.compute, k.out);
  });
}
SignLogDet slogdet(const ndarray& a) {
  require_square(a);
  SignLogDet out;
  dispatch(a.dtype(), [&](auto tag, Kind kk) -> ndarray {
    using T = decltype(tag);
    const int n = static_cast<int>(a.shape()[0]);
    std::vector<T> A = to_vec<T>(a, kk.compute);
    std::vector<int> piv; double sgn;
    DType real_out = kk.cmplx ? (kk.out == kComplex64 ? kFloat32 : kFloat64) : kk.out;
    if (!lu_decompose<T>(A, n, piv, sgn)) {
      std::vector<T> s{T(0)};
      out.sign = from_vec<T>(s, {}, kk.compute, kk.out);
      ndarray la(Shape{}, kFloat64); la.set_item<double>({}, -std::numeric_limits<double>::infinity());
      out.logabsdet = la.astype(real_out);
      return a;
    }
    double logabs = 0.0; T sign = T(sgn);
    for (int i = 0; i < n; ++i) {
      T d = A[i * n + i];
      double m = std::abs(d);
      logabs += std::log(m);
      sign *= d / T(m);
    }
    std::vector<T> s{sign};
    out.sign = from_vec<T>(s, {}, kk.compute, kk.out);
    ndarray la(Shape{}, kFloat64); la.set_item<double>({}, logabs);
    out.logabsdet = la.astype(real_out);
    return a;
  });
  return out;
}
ndarray matrix_power(const ndarray& a, int64_t n) {
  require_square(a);
  const int64_t d = a.shape()[0];
  if (n == 0) return eye(d, d, 0, la_kind(a.dtype()).out);
  ndarray base = n < 0 ? inv(a) : a.astype(la_kind(a.dtype()).out);
  int64_t e = n < 0 ? -n : n;
  ndarray result = eye(d, d, 0, base.dtype());
  while (e > 0) {                         // binary exponentiation
    if (e & 1) result = matmul(result, base);
    e >>= 1;
    if (e) base = matmul(base, base);
  }
  return result;
}
ndarray cholesky(const ndarray& a) {
  require_square(a);
  return dispatch(a.dtype(), [&](auto tag, Kind k) {
    using T = decltype(tag); return cholesky_impl<T>(a, k);
  });
}

}  // namespace linalg
}  // namespace numpp
