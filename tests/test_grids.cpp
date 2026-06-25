#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// ============================= meshgrid =============================
TEST_CASE("meshgrid xy X vs numpy") {
  ndarray x = arange(0., 4., 1., kFloat64);
  ndarray y = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.meshgrid(np.arange(4.),np.arange(3.))[0]");
  if (o) CHECK(allclose(meshgrid({x, y})[0], *o, 1e-9, 1e-12, true));
}
TEST_CASE("meshgrid xy Y vs numpy") {
  ndarray x = arange(0., 4., 1., kFloat64);
  ndarray y = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.meshgrid(np.arange(4.),np.arange(3.))[1]");
  if (o) CHECK(allclose(meshgrid({x, y})[1], *o, 1e-9, 1e-12, true));
}
TEST_CASE("meshgrid ij X vs numpy") {
  ndarray x = arange(0., 4., 1., kFloat64);
  ndarray y = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.meshgrid(np.arange(4.),np.arange(3.),indexing='ij')[0]");
  if (o) CHECK(allclose(meshgrid({x, y}, false)[0], *o, 1e-9, 1e-12, true));
}

// ============================= indices =============================
TEST_CASE("indices row vs numpy") {
  auto o = npt::oracle("a=np.indices((3,4))[0]");
  if (o) CHECK(allclose(indices({3, 4}).index({IndexItem{(int64_t)0}}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("indices col vs numpy") {
  auto o = npt::oracle("a=np.indices((3,4))[1]");
  if (o) CHECK(allclose(indices({3, 4}).index({IndexItem{(int64_t)1}}), *o, 1e-9, 1e-12, true));
}

// ============================= diag / diagflat =============================
TEST_CASE("diag build from 1d vs numpy") {
  ndarray v = arange(1., 4., 1., kFloat64);
  auto o = npt::oracle("a=np.diag(np.arange(1,4.))");
  if (o) CHECK(allclose(diag(v), *o, 1e-9, 1e-12, true));
}
TEST_CASE("diag build offset k1 vs numpy") {
  ndarray v = arange(1., 4., 1., kFloat64);
  auto o = npt::oracle("a=np.diag(np.arange(1,4.),k=1)");
  if (o) CHECK(allclose(diag(v, 1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("diag extract from 2d k-1 vs numpy") {
  ndarray m = arange(0., 9., 1., kFloat64).reshape({3, 3});
  auto o = npt::oracle("a=np.diag(np.arange(9.).reshape(3,3),k=-1)");
  if (o) CHECK(allclose(diag(m, -1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("diagflat vs numpy") {
  ndarray v = arange(1., 5., 1., kFloat64).reshape({2, 2});
  auto o = npt::oracle("a=np.diagflat(np.arange(1,5.).reshape(2,2))");
  if (o) CHECK(allclose(diagflat(v), *o, 1e-9, 1e-12, true));
}

// ============================= tri / tril / triu =============================
TEST_CASE("tri vs numpy") {
  auto o = npt::oracle("a=np.tri(4,3,k=0)");
  if (o) CHECK(allclose(tri(4, 3, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("tril vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.tril(np.arange(12.).reshape(3,4))");
  if (o) CHECK(allclose(tril(m), *o, 1e-9, 1e-12, true));
}
TEST_CASE("triu k1 vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.triu(np.arange(12.).reshape(3,4),k=1)");
  if (o) CHECK(allclose(triu(m, 1), *o, 1e-9, 1e-12, true));
}

// ============================= vander =============================
TEST_CASE("vander decreasing vs numpy") {
  ndarray x = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.vander(np.arange(1,5.))");
  if (o) CHECK(allclose(vander(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("vander increasing N3 vs numpy") {
  ndarray x = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.vander(np.arange(1,5.),N=3,increasing=True)");
  if (o) CHECK(allclose(vander(x, 3, true), *o, 1e-9, 1e-12, true));
}

// ============================= logspace / geomspace =============================
TEST_CASE("logspace vs numpy") {
  auto o = npt::oracle("a=np.logspace(0,3,4)");
  if (o) CHECK(allclose(logspace(0, 3, 4), *o, 1e-9, 1e-12, true));
}
TEST_CASE("logspace base2 vs numpy") {
  auto o = npt::oracle("a=np.logspace(1,4,4,base=2.0)");
  if (o) CHECK(allclose(logspace(1, 4, 4, true, 2.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("geomspace vs numpy") {
  auto o = npt::oracle("a=np.geomspace(1,1000,4)");
  if (o) CHECK(allclose(geomspace(1, 1000, 4), *o, 1e-9, 1e-12, true));
}

// ============================= fromfunction =============================
TEST_CASE("fromfunction i+j vs numpy") {
  auto o = npt::oracle("a=np.fromfunction(lambda i,j: i+j, (3,4))");
  if (o) CHECK(allclose(fromfunction([](const std::vector<int64_t>& idx) {
                          return static_cast<double>(idx[0] + idx[1]);
                        }, {3, 4}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("fromfunction i*10+j vs numpy") {
  auto o = npt::oracle("a=np.fromfunction(lambda i,j: i*10+j, (2,3))");
  if (o) CHECK(allclose(fromfunction([](const std::vector<int64_t>& idx) {
                          return static_cast<double>(idx[0] * 10 + idx[1]);
                        }, {2, 3}), *o, 1e-9, 1e-12, true));
}
