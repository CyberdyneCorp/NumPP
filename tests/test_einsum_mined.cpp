// Cases mined from numpy's numpy/_core/tests/test_einsum.py: diagonal/trace,
// transposes, reductions, matmul/dot/outer, ellipsis, batched and chained
// contractions — all replayed through the live-NumPy oracle.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("mined einsum: single-operand views and reductions vs numpy") {
  ndarray A = arange(0.0, 9.0, 1.0).reshape({3, 3});
  const char* PYA = "A=np.arange(9.).reshape(3,3); ";
  auto chk = [&](const std::string& sub, const std::string& py) {
    auto o = npt::oracle(std::string(PYA) + "a=np.array(np.einsum('" + py + "',A))");
    if (o) CHECK(allclose(einsum(sub, {A}), *o, 1e-12, 1e-12, true));
  };
  chk("ii", "ii");          // trace
  chk("ii->i", "ii->i");    // diagonal
  chk("ij->ji", "ij->ji");  // transpose
  chk("ij->i", "ij->i");    // row sum
  chk("ij->j", "ij->j");    // column sum
  chk("ij->", "ij->");      // full sum
}

TEST_CASE("mined einsum: three-index diagonal reduction iji->j vs numpy") {
  ndarray T = arange(0.0, 27.0, 1.0).reshape({3, 3, 3});
  auto o = npt::oracle("T=np.arange(27.).reshape(3,3,3); a=np.einsum('iji->j',T)");
  if (o) CHECK(allclose(einsum("iji->j", {T}), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined einsum: two-operand contractions vs numpy") {
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray B = arange(0.0, 6.0, 1.0).reshape({3, 2});
  ndarray u = arange(0.0, 3.0, 1.0);
  ndarray v = arange(3.0, 6.0, 1.0);
  const char* PY =
      "A=np.arange(6.).reshape(2,3); B=np.arange(6.).reshape(3,2); "
      "u=np.arange(3.); v=np.arange(3.,6.); ";
  auto chk = [&](const ndarray& got, const std::string& expr) {
    auto o = npt::oracle(std::string(PY) + "a=np.array(np.einsum(" + expr + "))");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(einsum("ij,jk->ik", {A, B}), "'ij,jk->ik',A,B");  // matmul
  chk(einsum("ij,jk", {A, B}), "'ij,jk',A,B");          // implicit output
  chk(einsum("i,i->", {u, v}), "'i,i->',u,v");          // dot
  chk(einsum("i,j->ij", {u, v}), "'i,j->ij',u,v");      // outer
  chk(einsum("ij,ij->", {A, A}), "'ij,ij->',A,A");      // full element-wise sum
}

TEST_CASE("mined einsum: ellipsis broadcasting vs numpy") {
  ndarray T = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  auto o = npt::oracle("T=np.arange(24.).reshape(2,3,4); a=np.einsum('...ij->...ji',T)");
  if (o) CHECK(allclose(einsum("...ij->...ji", {T}), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined einsum: batched matmul bij,bjk->bik vs numpy") {
  ndarray T = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  ndarray U = arange(0.0, 24.0, 1.0).reshape({2, 4, 3});
  auto o = npt::oracle(
      "T=np.arange(24.).reshape(2,3,4); U=np.arange(24.).reshape(2,4,3); "
      "a=np.einsum('bij,bjk->bik',T,U)");
  if (o) CHECK(allclose(einsum("bij,bjk->bik", {T, U}), *o, 1e-9, 1e-11, true));
}

TEST_CASE("mined einsum: chained three-operand contraction vs numpy") {
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray B = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray C = arange(0.0, 8.0, 1.0).reshape({4, 2});
  auto o = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(12.).reshape(3,4); "
      "C=np.arange(8.).reshape(4,2); a=np.einsum('ij,jk,kl->il',A,B,C)");
  if (o) CHECK(allclose(einsum("ij,jk,kl->il", {A, B, C}), *o, 1e-9, 1e-11, true));
}
