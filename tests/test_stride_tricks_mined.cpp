// Cases mined from numpy's numpy/lib/tests/test_stride_tricks.py:
// sliding_window_view (1-D and 2-D windows), broadcast_to, broadcast_arrays.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("mined stride_tricks: sliding_window_view 1-D vs numpy") {
  ndarray a = arange(0.0, 6.0, 1.0);
  ndarray w = sliding_window_view(a, {3});
  CHECK((w.shape() == Shape{4, 3}));
  auto o = npt::oracle(
      "a=np.lib.stride_tricks.sliding_window_view(np.arange(6.),3)");
  if (o) CHECK(allclose(w, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined stride_tricks: sliding_window_view 2-D window vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray w = sliding_window_view(A, {2, 2});
  CHECK((w.shape() == Shape{2, 3, 2, 2}));
  auto o = npt::oracle(
      "A=np.arange(12.).reshape(3,4); "
      "a=np.lib.stride_tricks.sliding_window_view(A,(2,2))");
  if (o) CHECK(allclose(w, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined stride_tricks: broadcast_to vs numpy") {
  ndarray a = arange(1.0, 4.0, 1.0);  // [1,2,3]
  ndarray b = a.broadcast_to(Shape{2, 3});
  auto o = npt::oracle("a=np.broadcast_to([1,2,3.],(2,3))");
  if (o) CHECK(allclose(b, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined stride_tricks: broadcast_arrays vs numpy") {
  ndarray x = arange(0.0, 3.0, 1.0);             // (3,)
  ndarray y = arange(0.0, 3.0, 1.0).reshape({3, 1});
  std::vector<ndarray> bc = broadcast_arrays({x, y});
  CHECK(bc.size() == 2);
  CHECK((bc[0].shape() == Shape{3, 3}));
  auto ox = npt::oracle("a=np.broadcast_arrays(np.arange(3.),np.arange(3.).reshape(3,1))[0]");
  if (ox) CHECK(allclose(bc[0], *ox, 1e-12, 1e-12, true));
  auto oy = npt::oracle("a=np.broadcast_arrays(np.arange(3.),np.arange(3.).reshape(3,1))[1]");
  if (oy) CHECK(allclose(bc[1], *oy, 1e-12, 1e-12, true));
}
