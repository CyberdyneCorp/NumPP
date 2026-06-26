#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("fromiter vs numpy") {
  std::vector<double> data{0.0, 1.5, -2.0, 3.25, 4.0};
  auto o = npt::oracle("a=np.fromiter([0.0,1.5,-2.0,3.25,4.0], dtype=np.float64)");
  if (o) CHECK(allclose(fromiter(data, kFloat64), *o, 1e-12, 1e-12, true));

  auto oi = npt::oracle("a=np.fromiter([0.0,1.5,-2.0,3.25,4.0], dtype=np.int32)");
  if (oi) CHECK(allclose(fromiter(data, kInt32), *oi, 1e-9, 1e-12, true));

  auto of = npt::oracle("a=np.fromiter([0.0,1.5,-2.0,3.25,4.0], dtype=np.float32)");
  if (of) CHECK(allclose(fromiter(data, kFloat32), *of, 1e-6, 1e-6, true));

  CHECK(fromiter(data).size() == 5);
  CHECK((fromiter(std::vector<double>{}).size() == 0));
}

TEST_CASE("frombuffer vs numpy") {
  // Build the same little-endian double bytes both sides use.
  std::string buf;
  double vals[4] = {0.0, 1.0, 2.0, 3.0};
  buf.resize(sizeof(vals));
  std::memcpy(buf.data(), vals, sizeof(vals));

  auto o = npt::oracle("a=np.frombuffer(np.arange(4.0).tobytes())");
  if (o) CHECK(allclose(frombuffer(buf, kFloat64), *o, 1e-12, 1e-12, true));

  // count limits the number of elements read.
  auto oc = npt::oracle("a=np.frombuffer(np.arange(4.0).tobytes(), count=2)");
  if (oc) CHECK(allclose(frombuffer(buf, kFloat64, 2), *oc, 1e-12, 1e-12, true));

  CHECK(frombuffer(buf, kFloat64).size() == 4);
  CHECK(frombuffer(buf, kFloat64, 2).size() == 2);
}

TEST_CASE("broadcast_arrays vs numpy") {
  ndarray a = arange(0.0, 3.0, 1.0, kFloat64).reshape({1, 3});
  ndarray b = arange(0.0, 2.0, 1.0, kFloat64).reshape({2, 1});
  std::vector<ndarray> res = broadcast_arrays({a, b});
  CHECK(res.size() == 2);
  CHECK((res[0].shape() == Shape{2, 3}));
  CHECK((res[1].shape() == Shape{2, 3}));

  auto o0 = npt::oracle(
      "x=np.arange(3.0).reshape(1,3); y=np.arange(2.0).reshape(2,1); a=np.broadcast_arrays(x,y)[0]");
  if (o0) CHECK(allclose(res[0], *o0, 1e-12, 1e-12, true));
  auto o1 = npt::oracle(
      "x=np.arange(3.0).reshape(1,3); y=np.arange(2.0).reshape(2,1); a=np.broadcast_arrays(x,y)[1]");
  if (o1) CHECK(allclose(res[1], *o1, 1e-12, 1e-12, true));
}

TEST_CASE("meshgrid_sparse vs numpy") {
  ndarray x = arange(0.0, 3.0, 1.0, kFloat64);
  ndarray y = arange(0.0, 4.0, 1.0, kFloat64);

  // Default xy indexing.
  std::vector<ndarray> xy = meshgrid_sparse({x, y}, true);
  CHECK(xy.size() == 2);
  auto ox0 = npt::oracle(
      "a=np.meshgrid(np.arange(3.0), np.arange(4.0), sparse=True, indexing='xy')[0]");
  if (ox0) CHECK(allclose(xy[0], *ox0, 1e-12, 1e-12, true));
  auto ox1 = npt::oracle(
      "a=np.meshgrid(np.arange(3.0), np.arange(4.0), sparse=True, indexing='xy')[1]");
  if (ox1) CHECK(allclose(xy[1], *ox1, 1e-12, 1e-12, true));

  // ij indexing keeps natural axis order.
  std::vector<ndarray> ij = meshgrid_sparse({x, y}, false);
  auto oi0 = npt::oracle(
      "a=np.meshgrid(np.arange(3.0), np.arange(4.0), sparse=True, indexing='ij')[0]");
  if (oi0) CHECK(allclose(ij[0], *oi0, 1e-12, 1e-12, true));
  auto oi1 = npt::oracle(
      "a=np.meshgrid(np.arange(3.0), np.arange(4.0), sparse=True, indexing='ij')[1]");
  if (oi1) CHECK(allclose(ij[1], *oi1, 1e-12, 1e-12, true));
}

TEST_CASE("mgrid vs numpy") {
  std::vector<ndarray> g = mgrid({{0.0, 3.0, 1.0}, {0.0, 4.0, 1.0}});
  CHECK(g.size() == 2);
  CHECK((g[0].shape() == Shape{3, 4}));

  auto o0 = npt::oracle("a=np.mgrid[0:3, 0:4][0].astype(np.float64)");
  if (o0) CHECK(allclose(g[0], *o0, 1e-12, 1e-12, true));
  auto o1 = npt::oracle("a=np.mgrid[0:3, 0:4][1].astype(np.float64)");
  if (o1) CHECK(allclose(g[1], *o1, 1e-12, 1e-12, true));
}

TEST_CASE("ogrid vs numpy") {
  std::vector<ndarray> g = ogrid({{0.0, 3.0, 1.0}, {0.0, 4.0, 1.0}});
  CHECK(g.size() == 2);
  CHECK((g[0].shape() == Shape{3, 1}));
  CHECK((g[1].shape() == Shape{1, 4}));

  auto o0 = npt::oracle("a=np.ogrid[0:3, 0:4][0].astype(np.float64)");
  if (o0) CHECK(allclose(g[0], *o0, 1e-12, 1e-12, true));
  auto o1 = npt::oracle("a=np.ogrid[0:3, 0:4][1].astype(np.float64)");
  if (o1) CHECK(allclose(g[1], *o1, 1e-12, 1e-12, true));
}
