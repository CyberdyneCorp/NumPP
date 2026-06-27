#include "numpp/linalg/batched.hpp"

#include <cstring>
#include <functional>
#include <vector>

#include "numpp/core/creation.hpp"

namespace numpp {
namespace linalg {
namespace batched {
namespace {

// A stack split into its leading (batch) shape and a (count, n, m) view.
struct Split {
  Shape lead;
  int64_t count;
  int64_t n, m;
  ndarray a3;  // contiguous, shape (count, n, m)
};

Split split(const ndarray& a) {
  const int64_t nd = a.ndim();
  Shape lead(a.shape().begin(), a.shape().end() - 2);
  int64_t count = 1;
  for (int64_t d : lead) count *= d;
  const int64_t n = a.shape()[nd - 2], m = a.shape()[nd - 1];
  ndarray a3 = a.ascontiguousarray().reshape({count, n, m});
  return {lead, count, n, m, a3};
}

// Stack equal-rank per-matrix results, promoting them to a common dtype first
// (so a stack with mixed real/complex eigenvalues becomes uniformly complex).
ndarray stack(const Shape& lead, std::vector<ndarray>& outs) {
  DType cd = outs[0].dtype();
  for (auto& o : outs) cd = result_type(cd, o.dtype());
  ndarray first = outs[0].astype(cd).ascontiguousarray();
  Shape full = lead;
  for (int64_t d : first.shape()) full.push_back(d);
  ndarray out(full, cd, Order::C);
  const int64_t slice_bytes = first.nbytes();
  char* dst = out.size() ? out.bytes() : nullptr;
  for (size_t b = 0; b < outs.size(); ++b) {
    ndarray rc = outs[b].astype(cd).ascontiguousarray();
    std::memcpy(dst + static_cast<int64_t>(b) * slice_bytes, rc.bytes(), slice_bytes);
  }
  return out;
}

// Apply a single-output 2-D op to every matrix and stack.
ndarray map1(const ndarray& a, const std::function<ndarray(const ndarray&)>& op) {
  Split s = split(a);
  std::vector<ndarray> outs;
  outs.reserve(static_cast<size_t>(s.count));
  for (int64_t b = 0; b < s.count; ++b) outs.push_back(op(s.a3[b].copy()));
  return stack(s.lead, outs);
}

// Apply a K-output 2-D op to every matrix and stack each output channel.
std::vector<ndarray> mapK(const ndarray& a, int K,
                          const std::function<std::vector<ndarray>(const ndarray&)>& op) {
  Split s = split(a);
  std::vector<std::vector<ndarray>> cols(static_cast<size_t>(K));
  for (int64_t b = 0; b < s.count; ++b) {
    std::vector<ndarray> rs = op(s.a3[b].copy());
    for (int k = 0; k < K; ++k) cols[static_cast<size_t>(k)].push_back(rs[static_cast<size_t>(k)]);
  }
  std::vector<ndarray> res;
  for (int k = 0; k < K; ++k) res.push_back(stack(s.lead, cols[static_cast<size_t>(k)]));
  return res;
}

}  // namespace

ndarray inv(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::inv(m); }); }
ndarray det(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::det(m); }); }
ndarray cholesky(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::cholesky(m); }); }
ndarray eigvals(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::eigvals(m); }); }
ndarray eigvalsh(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::eigvalsh(m); }); }
ndarray svdvals(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::svdvals(m); }); }
ndarray matrix_rank(const ndarray& a) { return map1(a, [](const ndarray& m) { return linalg::matrix_rank(m); }); }

ndarray matrix_power(const ndarray& a, int64_t n) {
  return map1(a, [n](const ndarray& m) { return linalg::matrix_power(m, n); });
}
ndarray pinv(const ndarray& a, double rcond) {
  return map1(a, [rcond](const ndarray& m) { return linalg::pinv(m, rcond); });
}

ndarray solve(const ndarray& a, const ndarray& b) {
  Split s = split(a);
  const bool bvec = b.ndim() == a.ndim() - 1;  // (..., n) vs (..., n, k)
  const int64_t k = bvec ? 1 : b.shape()[b.ndim() - 1];
  ndarray b3 = b.ascontiguousarray().reshape(bvec ? Shape{s.count, s.n} : Shape{s.count, s.n, k});
  std::vector<ndarray> outs;
  outs.reserve(static_cast<size_t>(s.count));
  for (int64_t i = 0; i < s.count; ++i)
    outs.push_back(linalg::solve(s.a3[i].copy(), b3[i].copy()));
  return stack(s.lead, outs);
}

SignLogDet slogdet(const ndarray& a) {
  auto r = mapK(a, 2, [](const ndarray& m) {
    SignLogDet sd = linalg::slogdet(m);
    return std::vector<ndarray>{sd.sign, sd.logabsdet};
  });
  return {r[0], r[1]};
}

QRResult qr(const ndarray& a, const std::string& mode) {
  auto r = mapK(a, 2, [&mode](const ndarray& m) {
    QRResult q = linalg::qr(m, mode);
    return std::vector<ndarray>{q.q, q.r};
  });
  return {r[0], r[1]};
}

EighResult eigh(const ndarray& a) {
  auto r = mapK(a, 2, [](const ndarray& m) {
    EighResult e = linalg::eigh(m);
    return std::vector<ndarray>{e.eigenvalues, e.eigenvectors};
  });
  return {r[0], r[1]};
}

EigResult eig(const ndarray& a) {
  auto r = mapK(a, 2, [](const ndarray& m) {
    EigResult e = linalg::eig(m);
    return std::vector<ndarray>{e.eigenvalues, e.eigenvectors};
  });
  return {r[0], r[1]};
}

SVDResult svd(const ndarray& a, bool full_matrices) {
  auto r = mapK(a, 3, [full_matrices](const ndarray& m) {
    SVDResult s = linalg::svd(m, full_matrices);
    return std::vector<ndarray>{s.u, s.s, s.vh};
  });
  return {r[0], r[1], r[2]};
}

}  // namespace batched
}  // namespace linalg
}  // namespace numpp
