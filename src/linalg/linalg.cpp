#include "numpp/linalg/linalg.hpp"

#include <algorithm>
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

// Householder QR: produces Q (m x qcols) and R (rrows x n), A = Q R.
template <class T>
void householder_qr(const std::vector<T>& Ain, int m, int n, bool complete,
                    std::vector<T>& Q, std::vector<T>& R, int& qcols, int& rrows) {
  std::vector<T> A = Ain;                       // becomes R
  std::vector<T> Qf(static_cast<size_t>(m) * m, T(0));
  for (int i = 0; i < m; ++i) Qf[i * m + i] = T(1);
  const int k = std::min(m, n);
  std::vector<T> v(m);
  for (int j = 0; j < k; ++j) {
    double xnorm = 0;
    for (int i = j; i < m; ++i) { double a = std::abs(A[i * n + j]); xnorm += a * a; }
    xnorm = std::sqrt(xnorm);
    if (xnorm == 0) continue;
    T a0 = A[j * n + j];
    T alpha;
    if constexpr (std::is_same_v<T, std::complex<double>>) {
      double r = std::abs(a0); alpha = -(r == 0 ? T(1) : a0 / T(r)) * T(xnorm);
    } else { alpha = (a0 >= 0 ? -xnorm : xnorm); }
    for (int i = 0; i < m; ++i) v[i] = T(0);
    for (int i = j; i < m; ++i) v[i] = A[i * n + j];
    v[j] -= alpha;
    double vn = 0; for (int i = j; i < m; ++i) { double a = std::abs(v[i]); vn += a * a; }
    if (vn == 0) continue;
    T beta = T(2.0 / vn);
    for (int col = j; col < n; ++col) {          // R <- H R
      T d(0); for (int i = j; i < m; ++i) d += conj_(v[i]) * A[i * n + col];
      d *= beta; for (int i = j; i < m; ++i) A[i * n + col] -= v[i] * d;
    }
    for (int row = 0; row < m; ++row) {          // Q <- Q H
      T d(0); for (int i = j; i < m; ++i) d += Qf[row * m + i] * v[i];
      d *= beta; for (int i = j; i < m; ++i) Qf[row * m + i] -= d * conj_(v[i]);
    }
  }
  if (complete) { qcols = m; rrows = m; Q = Qf; R = A; }
  else {
    qcols = k; rrows = k;
    Q.assign(static_cast<size_t>(m) * k, T(0));
    for (int i = 0; i < m; ++i) for (int c = 0; c < k; ++c) Q[i * k + c] = Qf[i * m + c];
    R.assign(static_cast<size_t>(k) * n, T(0));
    for (int i = 0; i < k; ++i) for (int c = 0; c < n; ++c) R[i * n + c] = A[i * n + c];
  }
}

// Cyclic Jacobi for a real symmetric matrix -> eigenvalues (ascending) + eigenvectors (columns).
void jacobi_real(std::vector<double> A, int n, std::vector<double>& eval, std::vector<double>& V) {
  V.assign(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0;
    for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-30) break;
    for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
      double apq = A[p * n + q];
      if (std::abs(apq) < 1e-300) continue;
      double theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
      double c = 1 / std::sqrt(t * t + 1), s = t * c;
      for (int i = 0; i < n; ++i) { double a = A[i * n + p], b = A[i * n + q]; A[i * n + p] = c * a - s * b; A[i * n + q] = s * a + c * b; }
      for (int i = 0; i < n; ++i) { double a = A[p * n + i], b = A[q * n + i]; A[p * n + i] = c * a - s * b; A[q * n + i] = s * a + c * b; }
      for (int i = 0; i < n; ++i) { double a = V[i * n + p], b = V[i * n + q]; V[i * n + p] = c * a - s * b; V[i * n + q] = s * a + c * b; }
    }
  }
  eval.assign(n, 0.0);
  std::vector<int> ord(n);
  for (int i = 0; i < n; ++i) { eval[i] = A[i * n + i]; ord[i] = i; }
  std::sort(ord.begin(), ord.end(), [&](int x, int y) { return eval[x] < eval[y]; });
  std::vector<double> es(n); std::vector<double> Vs(static_cast<size_t>(n) * n);
  for (int c = 0; c < n; ++c) { es[c] = eval[ord[c]]; for (int i = 0; i < n; ++i) Vs[i * n + c] = V[i * n + ord[c]]; }
  eval.swap(es); V.swap(Vs);
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

template <class T>
QRResult qr_impl(const ndarray& a, bool complete, Kind k) {
  const int m = static_cast<int>(a.shape()[0]), n = static_cast<int>(a.shape()[1]);
  std::vector<T> A = to_vec<T>(a, k.compute), Q, R;
  int qc = 0, rr = 0;
  householder_qr<T>(A, m, n, complete, Q, R, qc, rr);
  QRResult out;
  out.q = from_vec<T>(Q, {static_cast<int64_t>(m), static_cast<int64_t>(qc)}, k.compute, k.out);
  out.r = from_vec<T>(R, {static_cast<int64_t>(rr), static_cast<int64_t>(n)}, k.compute, k.out);
  return out;
}
QRResult qr(const ndarray& a, const std::string& mode) {
  if (a.ndim() != 2) throw not_implemented_error("qr requires a 2-D array");
  if (mode != "reduced" && mode != "complete") throw value_error("qr mode must be 'reduced' or 'complete'");
  Kind k = la_kind(a.dtype());
  if (k.cmplx) return qr_impl<std::complex<double>>(a, mode == "complete", k);
  return qr_impl<double>(a, mode == "complete", k);
}

namespace {
// Build the real symmetric problem (real input as-is; complex Hermitian via the
// 2n x 2n real embedding) and return ascending eigenvalues + (optionally) vectors.
void hermitian_eig(const ndarray& a, std::vector<double>& eval, ndarray* evec_out, Kind k) {
  const int n = static_cast<int>(a.shape()[0]);
  if (!k.cmplx) {
    std::vector<double> A = to_vec<double>(a, kFloat64), V;
    jacobi_real(A, n, eval, V);
    if (evec_out) *evec_out = from_vec<double>(V, {(int64_t)n, (int64_t)n}, kFloat64, k.out);
    return;
  }
  std::vector<std::complex<double>> A = to_vec<std::complex<double>>(a, kComplex128);
  std::vector<double> M(static_cast<size_t>(2 * n) * 2 * n, 0.0);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
    double re = A[i * n + j].real(), im = A[i * n + j].imag();
    M[i * 2 * n + j] = re;            M[i * 2 * n + (n + j)] = -im;
    M[(n + i) * 2 * n + j] = im;      M[(n + i) * 2 * n + (n + j)] = re;
  }
  std::vector<double> e2, V2;
  jacobi_real(M, 2 * n, e2, V2);
  eval.resize(n);
  for (int c = 0; c < n; ++c) eval[c] = e2[2 * c];          // pairs are identical
  if (evec_out) {
    std::vector<std::complex<double>> Z(static_cast<size_t>(n) * n);
    for (int c = 0; c < n; ++c) {
      double nrm = 0;
      for (int i = 0; i < n; ++i) { std::complex<double> z(V2[i * 2 * n + 2 * c], V2[(n + i) * 2 * n + 2 * c]); Z[i * n + c] = z; nrm += std::norm(z); }
      nrm = std::sqrt(nrm);
      if (nrm > 0) for (int i = 0; i < n; ++i) Z[i * n + c] /= nrm;
    }
    *evec_out = from_vec<std::complex<double>>(Z, {(int64_t)n, (int64_t)n}, kComplex128, k.out);
  }
}
ndarray real_evals(const std::vector<double>& eval, DType real_out) {
  ndarray r(Shape{static_cast<int64_t>(eval.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < eval.size(); ++i) r.set_item<double>({(int64_t)i}, eval[i]);
  return r.astype(real_out);
}
}  // namespace

ndarray eigvalsh(const ndarray& a) {
  require_square(a);
  Kind k = la_kind(a.dtype());
  std::vector<double> eval;
  hermitian_eig(a, eval, nullptr, k);
  DType real_out = (k.out == kComplex64 || k.out == kFloat32) ? kFloat32 : kFloat64;
  return real_evals(eval, real_out);
}
EighResult eigh(const ndarray& a) {
  require_square(a);
  Kind k = la_kind(a.dtype());
  std::vector<double> eval; ndarray evec;
  hermitian_eig(a, eval, &evec, k);
  DType real_out = (k.out == kComplex64 || k.out == kFloat32) ? kFloat32 : kFloat64;
  return {real_evals(eval, real_out), evec};
}

namespace {
// Orthonormal-complete `have` columns (m x have) to `want` columns via modified
// Gram-Schmidt against the standard basis (preserves the original columns).
template <class T>
std::vector<T> gram_schmidt_extend(const std::vector<T>& cols, int m, int have, int want) {
  std::vector<std::vector<T>> basis;
  for (int c = 0; c < have; ++c) { std::vector<T> v(m); for (int i = 0; i < m; ++i) v[i] = cols[i * have + c]; basis.push_back(v); }
  for (int e = 0; e < m && (int)basis.size() < want; ++e) {
    std::vector<T> v(m, T(0)); v[e] = T(1);
    for (auto& b : basis) { T d(0); for (int i = 0; i < m; ++i) d += conj_(b[i]) * v[i]; for (int i = 0; i < m; ++i) v[i] -= d * b[i]; }
    double nrm = 0; for (int i = 0; i < m; ++i) { double a = std::abs(v[i]); nrm += a * a; } nrm = std::sqrt(nrm);
    if (nrm > 1e-9) { for (int i = 0; i < m; ++i) v[i] /= T(nrm); basis.push_back(v); }
  }
  std::vector<T> out(static_cast<size_t>(m) * want, T(0));
  for (int c = 0; c < (int)basis.size() && c < want; ++c) for (int i = 0; i < m; ++i) out[i * want + c] = basis[c][i];
  return out;
}

template <class T>
SVDResult svd_impl(const ndarray& a, bool full_matrices, Kind k) {
  const int m = static_cast<int>(a.shape()[0]), n = static_cast<int>(a.shape()[1]), kk = std::min(m, n);
  const bool wide = m < n;
  std::vector<T> A = to_vec<T>(a, k.compute);
  const int gN = wide ? m : n;
  std::vector<T> G(static_cast<size_t>(gN) * gN, T(0));
  if (!wide) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { T s(0); for (int p = 0; p < m; ++p) s += conj_(A[p * n + i]) * A[p * n + j]; G[i * gN + j] = s; }
  else       for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) { T s(0); for (int p = 0; p < n; ++p) s += A[i * n + p] * conj_(A[j * n + p]); G[i * gN + j] = s; }
  ndarray Gnd = from_vec<T>(G, {(int64_t)gN, (int64_t)gN}, k.compute, k.compute);
  std::vector<double> ev; ndarray evec;
  hermitian_eig(Gnd, ev, &evec, la_kind(k.compute));
  std::vector<T> W = to_vec<T>(evec, k.compute);  // gN x gN, columns ascending
  std::vector<double> sd(kk);
  for (int i = 0; i < kk; ++i) { double e = ev[gN - 1 - i]; sd[i] = e > 0 ? std::sqrt(e) : 0.0; }

  std::vector<T> Uout, Vout;
  int ucols = full_matrices ? m : kk;
  int vcols;
  if (!wide) {
    std::vector<T> V(static_cast<size_t>(n) * n);                  // n x n (kk == n)
    for (int c = 0; c < n; ++c) for (int i = 0; i < n; ++i) V[i * n + c] = W[i * gN + (gN - 1 - c)];
    std::vector<T> U(static_cast<size_t>(m) * kk, T(0));
    for (int c = 0; c < kk; ++c) if (sd[c] > 0) for (int i = 0; i < m; ++i) { T s(0); for (int p = 0; p < n; ++p) s += A[i * n + p] * V[p * n + c]; U[i * kk + c] = s / T(sd[c]); }
    Uout = full_matrices ? gram_schmidt_extend<T>(U, m, kk, m) : U;
    Vout = V; vcols = n;
  } else {
    std::vector<T> U(static_cast<size_t>(m) * m);                  // m x m (kk == m)
    for (int c = 0; c < m; ++c) for (int i = 0; i < m; ++i) U[i * m + c] = W[i * gN + (gN - 1 - c)];
    std::vector<T> V(static_cast<size_t>(n) * kk, T(0));
    for (int c = 0; c < kk; ++c) if (sd[c] > 0) for (int i = 0; i < n; ++i) { T s(0); for (int p = 0; p < m; ++p) s += conj_(A[p * n + i]) * U[p * m + c]; V[i * kk + c] = s / T(sd[c]); }
    if (full_matrices) { Uout = U; Vout = gram_schmidt_extend<T>(V, n, kk, n); vcols = n; }
    else { std::vector<T> Ur(static_cast<size_t>(m) * kk); for (int i = 0; i < m; ++i) for (int c = 0; c < kk; ++c) Ur[i * kk + c] = U[i * m + c]; Uout = Ur; Vout = V; vcols = kk; }
  }
  std::vector<T> Vh(static_cast<size_t>(vcols) * n);
  for (int r = 0; r < vcols; ++r) for (int j = 0; j < n; ++j) Vh[r * n + j] = conj_(Vout[j * vcols + r]);

  SVDResult out;
  out.u = from_vec<T>(Uout, {(int64_t)m, (int64_t)ucols}, k.compute, k.out);
  DType real_out = (k.out == kComplex64 || k.out == kFloat32) ? kFloat32 : kFloat64;
  ndarray sArr(Shape{(int64_t)kk}, kFloat64, Order::C);
  for (int i = 0; i < kk; ++i) sArr.set_item<double>({i}, sd[i]);
  out.s = sArr.astype(real_out);
  out.vh = from_vec<T>(Vh, {(int64_t)vcols, (int64_t)n}, k.compute, k.out);
  return out;
}

double dtype_eps(DType d) { return (d == kFloat32) ? 1.1920928955078125e-7 : 2.220446049250313e-16; }
}  // namespace

SVDResult svd(const ndarray& a, bool full_matrices) {
  if (a.ndim() != 2) throw not_implemented_error("svd requires a 2-D array");
  Kind k = la_kind(a.dtype());
  if (k.cmplx) return svd_impl<std::complex<double>>(a, full_matrices, k);
  return svd_impl<double>(a, full_matrices, k);
}
ndarray svdvals(const ndarray& a) { return svd(a, false).s; }

ndarray pinv(const ndarray& a, double rcond) {
  SVDResult r = svd(a, /*full_matrices=*/false);
  const int64_t kk = r.s.shape()[0];
  ndarray smax = amax(r.s);
  double smax_d = smax.astype(kFloat64).item<double>({});
  double tol = rcond * smax_d;
  ndarray sd = r.s.astype(kFloat64);
  ndarray sinv(Shape{kk}, kFloat64, Order::C);
  for (int64_t i = 0; i < kk; ++i) { double s = sd.item<double>({i}); sinv.set_item<double>({i}, s > tol ? 1.0 / s : 0.0); }
  ndarray V = conj(r.vh).transpose();              // n x kk
  ndarray Uh = conj(r.u).transpose();              // kk x m
  ndarray scaled = multiply(Uh, sinv.reshape({kk, 1}));
  return matmul(V, scaled).astype(la_kind(a.dtype()).out);
}

ndarray matrix_rank(const ndarray& a) {
  ndarray s = svdvals(a).astype(kFloat64);
  const int64_t kk = s.shape()[0];
  double smax = kk ? amax(s).item<double>({}) : 0.0;
  double tol = smax * std::max<int64_t>(a.shape()[0], a.shape()[1]) * dtype_eps(la_kind(a.dtype()).out);
  int64_t rank = 0;
  for (int64_t i = 0; i < kk; ++i) if (s.item<double>({i}) > tol) ++rank;
  ndarray out(Shape{}, kInt64, Order::C); out.set_item<int64_t>({}, rank);
  return out;
}

LstsqResult lstsq(const ndarray& a, const ndarray& b, double rcond) {
  SVDResult r = svd(a, /*full_matrices=*/false);
  const int m = static_cast<int>(a.shape()[0]), n = static_cast<int>(a.shape()[1]);
  const int64_t kk = r.s.shape()[0];
  ndarray sd = r.s.astype(kFloat64);
  double smax = kk ? amax(sd).item<double>({}) : 0.0;
  double tol = (rcond < 0 ? dtype_eps(la_kind(a.dtype()).out) * std::max(m, n) : rcond) * smax;
  ndarray sinv(Shape{kk}, kFloat64, Order::C);
  int64_t rank = 0;
  for (int64_t i = 0; i < kk; ++i) { double s = sd.item<double>({i}); bool ok = s > tol; sinv.set_item<double>({i}, ok ? 1.0 / s : 0.0); rank += ok; }
  const bool bvec = b.ndim() == 1;
  ndarray bm = bvec ? b.reshape({b.size(), 1}) : b;
  ndarray Uh = conj(r.u).transpose();              // kk x m
  ndarray V = conj(r.vh).transpose();              // n x kk
  ndarray x = matmul(V, multiply(sinv.reshape({kk, 1}), matmul(Uh, bm.astype(r.u.dtype()))));
  LstsqResult out;
  out.solution = bvec ? x.reshape({n}) : x;
  out.solution = out.solution.astype(la_kind(result_type(a.dtype(), b.dtype())).out);
  ndarray rk(Shape{}, kInt64, Order::C); rk.set_item<int64_t>({}, rank); out.rank = rk;
  out.singular_values = r.s;
  // residuals: only when overdetermined and full column rank
  if (m > n && rank == n) {
    ndarray resid = matmul(a.astype(out.solution.dtype()), bvec ? out.solution.reshape({n, 1}) : out.solution);
    resid = subtract(bm.astype(out.solution.dtype()), resid);
    out.residuals = sum(square(absolute(resid)), 0);
  } else {
    out.residuals = ndarray(Shape{0}, kFloat64, Order::C);
  }
  return out;
}

namespace {
ndarray norm_vector(const ndarray& a, int which, double p) {
  ndarray av = absolute(a);                       // real magnitudes
  switch (which) {
    case 0: return sqrt(sum(square(av)));          // 2-norm (default)
    case 1: return sum(av);                        // 1-norm
    case 2: return amax(av);                       // inf
    case 3: return amin(av);                       // -inf
    case 4: return sum(not_equal(a, scalar_like(0.0, a.dtype(), false))).astype(kFloat64);  // 0
    default: {                                     // general p
      ndarray s = sum(power(av, scalar_like(p, kFloat64, true)));
      return power(s, scalar_like(1.0 / p, kFloat64, true));
    }
  }
}
ndarray norm_matrix(const ndarray& a, int which) {
  ndarray av = absolute(a);
  switch (which) {
    case 0: return sqrt(sum(square(av)));          // fro (default)
    case 1: return amax(sum(av, 0));               // 1
    case 2: return amax(sum(av, 1));               // inf
    case -1: return amin(sum(av, 0));
    case -2: return amin(sum(av, 1));
    case 3: return amax(svdvals(a));     // spectral (ord 2)
    case -3: return amin(svdvals(a));    // ord -2
    case 4: return sum(svdvals(a));      // nuclear
    default: throw not_implemented_error("unsupported matrix norm order");
  }
}
}  // namespace

ndarray norm(const ndarray& a) {
  return a.ndim() == 1 ? norm_vector(a, 0, 0) : norm_matrix(a, 0);
}
ndarray norm(const ndarray& a, double ord) {
  const double inf = std::numeric_limits<double>::infinity();
  if (a.ndim() == 1) {
    if (ord == 1) return norm_vector(a, 1, 0);
    if (ord == 2) return norm_vector(a, 0, 0);
    if (ord == inf) return norm_vector(a, 2, 0);
    if (ord == -inf) return norm_vector(a, 3, 0);
    if (ord == 0) return norm_vector(a, 4, 0);
    return norm_vector(a, 99, ord);
  }
  if (ord == 1) return norm_matrix(a, 1);
  if (ord == inf) return norm_matrix(a, 2);
  if (ord == -1) return norm_matrix(a, -1);
  if (ord == -inf) return norm_matrix(a, -2);
  if (ord == 2) return norm_matrix(a, 3);
  if (ord == -2) return norm_matrix(a, -3);
  return norm_matrix(a, 99);
}
ndarray norm(const ndarray& a, const std::string& ord) {
  if (ord == "fro") return norm_matrix(a, 0);
  if (ord == "nuc") return norm_matrix(a, 4);
  throw not_implemented_error("norm ord '" + ord + "' not supported");
}

}  // namespace linalg
}  // namespace numpp
