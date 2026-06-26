#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("boolean_index vs numpy") {
  ndarray x = arange(0., 10., 1., kFloat64);
  ndarray mask = greater(x, full(x.shape(), 4., kFloat64));
  auto o = npt::oracle("a=np.arange(10.); a=a[a>4]");
  if (o) CHECK(allclose(boolean_index(x, mask), *o, 1e-9, 1e-12, true));
}

TEST_CASE("boolean_index 2d broadcast mask vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray mask = greater(x, full(x.shape(), 5., kFloat64));
  auto o = npt::oracle("a=np.arange(12.).reshape(3,4); a=a[a>5]");
  if (o) CHECK(allclose(boolean_index(x, mask), *o, 1e-9, 1e-12, true));
}

TEST_CASE("boolean_assign cycles values vs numpy") {
  ndarray x = arange(0., 10., 1., kFloat64);
  ndarray mask = greater(x, full(x.shape(), 4., kFloat64));
  boolean_assign(x, mask, full({1}, 0., kFloat64));
  auto o = npt::oracle("a=np.arange(10.); a[a>4]=0.");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("fancy_index flat vs numpy") {
  ndarray x = arange(0., 10., 1., kFloat64);
  ndarray idx = arange(2., 7., 2., kInt64);  // [2,4,6]
  auto o = npt::oracle("a=np.arange(10.)[[2,4,6]]");
  if (o) CHECK(allclose(fancy_index(x, idx), *o, 1e-9, 1e-12, true));
}

TEST_CASE("fancy_assign vs numpy") {
  ndarray x = arange(0., 10., 1., kFloat64);
  ndarray idx = arange(1., 6., 2., kInt64);  // [1,3,5]
  ndarray vals({3}, kFloat64, Order::C);
  vals.set_item<double>({0}, 100.);
  vals.set_item<double>({1}, 200.);
  vals.set_item<double>({2}, 300.);
  fancy_assign(x, idx, vals);
  auto o = npt::oracle("a=np.arange(10.); a[[1,3,5]]=[100.,200.,300.]");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("put_along_axis vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray idx = arange(0., 3., 1., kInt64).reshape({3, 1});  // [[0],[1],[2]]
  put_along_axis(x, idx, full({1}, 99., kFloat64), 1);
  auto o = npt::oracle(
      "a=np.arange(12.).reshape(3,4); "
      "np.put_along_axis(a, np.array([[0],[1],[2]]), 99., axis=1)");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("place cycles vals vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray mask = greater(x, full(x.shape(), 2., kFloat64));
  ndarray vals({2}, kFloat64, Order::C);
  vals.set_item<double>({0}, 11.);
  vals.set_item<double>({1}, 22.);
  place(x, mask, vals);
  auto o = npt::oracle(
      "a=np.arange(6.).reshape(2,3); np.place(a, a>2, [11.,22.])");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("ix_ first array vs numpy") {
  ndarray r = arange(0., 3., 2., kInt64);  // [0,2]
  ndarray c = arange(1., 4., 2., kInt64);  // [1,3]
  auto mesh = ix_({r, c});
  auto o0 = npt::oracle("a=np.ix_([0,2],[1,3])[0]");
  if (o0) CHECK(allclose(mesh[0], *o0, 1e-9, 1e-12, true));
  auto o1 = npt::oracle("a=np.ix_([0,2],[1,3])[1]");
  if (o1) CHECK(allclose(mesh[1], *o1, 1e-9, 1e-12, true));
}

TEST_CASE("fill_diagonal vs numpy") {
  ndarray x = zeros({3, 3}, kFloat64);
  fill_diagonal(x, 5.);
  auto o = npt::oracle("a=np.zeros((3,3)); np.fill_diagonal(a, 5.)");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("fill_diagonal non-square vs numpy") {
  ndarray x = zeros({2, 4}, kFloat64);
  fill_diagonal(x, 7.);
  auto o = npt::oracle("a=np.zeros((2,4)); np.fill_diagonal(a, 7.)");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-12, true));
}

TEST_CASE("diag_indices vs numpy") {
  auto di = diag_indices(4);
  auto o = npt::oracle("a=np.diag_indices(4)[0]");
  if (o) CHECK(allclose(di[0], *o, 1e-9, 1e-12, true));
}

TEST_CASE("tril_indices vs numpy") {
  auto t = tril_indices(4, -1);
  auto o0 = npt::oracle("a=np.tril_indices(4,-1)[0]");
  if (o0) CHECK(allclose(t[0], *o0, 1e-9, 1e-12, true));
  auto o1 = npt::oracle("a=np.tril_indices(4,-1)[1]");
  if (o1) CHECK(allclose(t[1], *o1, 1e-9, 1e-12, true));
}

TEST_CASE("triu_indices vs numpy") {
  auto t = triu_indices(3);
  auto o = npt::oracle("a=np.triu_indices(3)[1]");
  if (o) CHECK(allclose(t[1], *o, 1e-9, 1e-12, true));
}
