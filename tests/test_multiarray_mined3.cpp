// Deeper test_multiarray.py mining: reduction dtype upcasting + integer overflow,
// astype casting, partition/argpartition, searchsorted sides, and axis sort.
#include "numpp/numpp.hpp"
#include "numpp/sorting/sorting.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <complex>
#include <vector>

using namespace numpp;
using cd = std::complex<double>;

namespace {
ndarray i8(const std::vector<int8_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt8, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int8_t>()[i] = v[i];
  return a;
}
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray cval(const std::vector<cd>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<cd>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined multiarray: sum/prod of int8 upcasts (no overflow) vs numpy") {
  ndarray a = i8({100, 100, 100});  // 300 overflows int8 but numpy sums in int64
  auto os = npt::oracle("a=np.array(np.sum(np.array([100,100,100],dtype=np.int8)))");
  if (os) CHECK(allclose(sum(a).astype(kFloat64), os->astype(kFloat64), 0, 0, true));
  ndarray b = i8({5, 6, 7});  // 210 fits int8 but product 210 < 128? 5*6*7=210 overflows int8
  auto op = npt::oracle("a=np.array(np.prod(np.array([5,6,7],dtype=np.int8)))");
  if (op) CHECK(allclose(prod(b).astype(kFloat64), op->astype(kFloat64), 0, 0, true));
}

TEST_CASE("mined multiarray: mean of an integer array is float vs numpy") {
  ndarray a = i8({1, 2, 3, 4});
  auto o = npt::oracle("a=np.array(np.mean(np.array([1,2,3,4],dtype=np.int8)))");
  if (o) CHECK(allclose(mean(a), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: integer add/multiply overflow wraps vs numpy") {
  ndarray a = i8({127, -128, 100});
  ndarray b = i8({1, -1, 100});
  auto oa = npt::oracle(
      "a=(np.array([127,-128,100],dtype=np.int8)+np.array([1,-1,100],dtype=np.int8))");
  auto om = npt::oracle(
      "a=(np.array([127,-128,100],dtype=np.int8)*np.array([1,-1,100],dtype=np.int8))");
  if (oa) CHECK(allclose(add(a, b).astype(kFloat64), oa->astype(kFloat64), 0, 0, true));
  if (om) CHECK(allclose(multiply(a, b).astype(kFloat64), om->astype(kFloat64), 0, 0, true));
}

TEST_CASE("mined multiarray: astype casting (float->int trunc, complex->real) vs numpy") {
  ndarray f = dval({-2.7, -1.2, 0.9, 2.5, 3.99});
  auto oi = npt::oracle("a=np.array([-2.7,-1.2,0.9,2.5,3.99]).astype(np.int64)");
  if (oi) CHECK(allclose(f.astype(kInt64).astype(kFloat64), oi->astype(kFloat64), 0, 0, true));
  ndarray z = cval({cd(1, 2), cd(-3, 4), cd(5, -6)});
  auto oc = npt::oracle("a=np.array([1+2j,-3+4j,5-6j]).astype(np.float64)");  // takes real part
  if (oc) CHECK(allclose(z.astype(kFloat64), *oc, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: partition places the kth element vs numpy") {
  ndarray a = dval({7, 2, 9, 1, 5, 3, 8});
  ndarray got = partition(a, 3);
  // The element at index kth equals the sorted element; everything left is <=, right >=.
  auto o = npt::oracle("a=np.partition([7,2,9,1,5,3,8.],3)");
  if (o) {
    CHECK(got.item<double>({3}) == o->item<double>({3}));
    // left partition <= pivot, right >= pivot
    double piv = got.item<double>({3});
    for (int i = 0; i < 3; ++i) CHECK(got.item<double>({static_cast<int64_t>(i)}) <= piv);
    for (int i = 4; i < 7; ++i) CHECK(got.item<double>({static_cast<int64_t>(i)}) >= piv);
  }
}

TEST_CASE("mined multiarray: argpartition kth vs numpy") {
  ndarray a = dval({7, 2, 9, 1, 5, 3, 8});
  ndarray idx = argpartition(a, 3);
  // a[idx[3]] must equal the kth smallest value.
  auto o = npt::oracle("a=np.array([np.partition([7,2,9,1,5,3,8.],3)[3]])");
  if (o) CHECK(a.item<double>({idx.item<int64_t>({3})}) == o->item<double>({0}));
}

TEST_CASE("mined multiarray: searchsorted side left/right vs numpy") {
  ndarray a = dval({1, 2, 2, 2, 3, 5});
  ndarray v = dval({2, 4, 0, 5});
  auto ol = npt::oracle("a=np.searchsorted([1,2,2,2,3,5.],[2,4,0,5.],side='left')");
  auto orr = npt::oracle("a=np.searchsorted([1,2,2,2,3,5.],[2,4,0,5.],side='right')");
  if (ol) CHECK(allclose(searchsorted(a, v, "left").astype(kFloat64), ol->astype(kFloat64), 0, 0, true));
  if (orr) CHECK(allclose(searchsorted(a, v, "right").astype(kFloat64), orr->astype(kFloat64), 0, 0, true));
}

TEST_CASE("mined multiarray: sort along an axis vs numpy") {
  ndarray A = dval({3, 1, 2, 9, 7, 8, 0, 5, 4}).reshape({3, 3});
  auto o0 = npt::oracle("A=np.array([[3,1,2],[9,7,8],[0,5,4.]]); a=np.sort(A,axis=0)");
  auto o1 = npt::oracle("A=np.array([[3,1,2],[9,7,8],[0,5,4.]]); a=np.sort(A,axis=1)");
  if (o0) CHECK(allclose(sort(A, int64_t{0}), *o0, 0, 0, true));
  if (o1) CHECK(allclose(sort(A, int64_t{1}), *o1, 0, 0, true));
}
