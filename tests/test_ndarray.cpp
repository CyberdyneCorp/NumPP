#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static ndarray iota(const Shape& shape, DType dt = kFloat64) {
  ndarray a(shape, dt, Order::C);
  double v = 0;
  a.for_each_offset([&](int64_t off) {
    double d = v++;
    std::memcpy(a.buffer()->data() + off, &d, sizeof(double));
  });
  return a;
}

TEST_CASE("construction and metadata") {
  ndarray a({2, 3}, kFloat64);
  CHECK(a.ndim() == 2);
  CHECK(a.size() == 6);
  CHECK(a.itemsize() == 8);
  CHECK(a.c_contiguous());
  CHECK(a.strides()[0] == 24);
  CHECK(a.strides()[1] == 8);
  ndarray scalar({}, kFloat64);
  CHECK(scalar.ndim() == 0);
  CHECK(scalar.size() == 1);
}

TEST_CASE("view shares storage, copy is independent") {
  ndarray a = iota({4});
  ndarray b = a[1];                   // view of element 1 (0-d)
  CHECK(b.ndim() == 0);
  ndarray sl = a.index({Slice{1, 3, 1}});
  sl.set_item<double>({0}, 99.0);
  CHECK(a.item<double>({1}) == 99.0); // view writes through
  ndarray c = a.copy();
  c.set_item<double>({1}, 5.0);
  CHECK(a.item<double>({1}) == 99.0); // copy independent
}

TEST_CASE("lifetime via shared buffer") {
  ndarray b;
  {
    ndarray a = iota({3});
    b = a.index({Slice{0, 3, 1}});
  }  // a destroyed
  CHECK(b.item<double>({2}) == 2.0);
}

TEST_CASE("transpose toggles contiguity and shares buffer") {
  ndarray a = iota({2, 3});
  ndarray t = a.transpose();
  CHECK(t.shape()[0] == 3 && t.shape()[1] == 2);
  CHECK(t.f_contiguous());
  CHECK(!t.c_contiguous());
  CHECK(t.buffer().get() == a.buffer().get());
  CHECK(t.item<double>({2, 1}) == a.item<double>({1, 2}));
}

TEST_CASE("reshape view, inference, error") {
  ndarray a = iota({2, 6});
  ndarray r = a.reshape({3, 4});
  CHECK(r.shape()[0] == 3 && r.shape()[1] == 4);
  CHECK(r.buffer().get() == a.buffer().get());
  ndarray inf = iota({12}).reshape({3, -1});
  CHECK(inf.shape()[1] == 4);
  CHECK_THROWS_AS(iota({12}).reshape({5, 5}), value_error);
}

TEST_CASE("squeeze and expand_dims") {
  ndarray a({1, 3, 1}, kFloat64);
  CHECK(a.squeeze().ndim() == 1);
  CHECK(a.squeeze().shape()[0] == 3);
  ndarray e = iota({3, 4}).expand_dims(1);
  CHECK(e.shape()[0] == 3 && e.shape()[1] == 1 && e.shape()[2] == 4);
  CHECK(iota({3}).expand_dims(-1).shape()[1] == 1);
}

TEST_CASE("strided and reversed slices") {
  ndarray a = iota({6});
  ndarray s = a.index({Slice{1, 5, 2}});
  CHECK(s.size() == 2);
  CHECK(s.item<double>({0}) == 1.0 && s.item<double>({1}) == 3.0);
  ndarray rev = iota({4}).index({Slice{std::nullopt, std::nullopt, -1}});
  CHECK(rev.item<double>({0}) == 3.0 && rev.item<double>({3}) == 0.0);
}

TEST_CASE("ravel view vs flatten copy") {
  ndarray a = iota({2, 3});
  ndarray r = a.ravel();
  CHECK(r.ndim() == 1 && r.buffer().get() == a.buffer().get());
  ndarray t = a.transpose();             // non-C-contiguous
  ndarray rc = t.ravel();                // must copy
  CHECK(rc.buffer().get() != t.buffer().get());
  ndarray fl = a.flatten();
  fl.set_item<double>({0}, 42.0);
  CHECK(a.item<double>({0, 0}) == 0.0);  // flatten independent
}

TEST_CASE("broadcast_to is read-only zero-stride view") {
  ndarray a = iota({3, 1});
  ndarray b = a.broadcast_to({3, 4});
  CHECK(b.shape()[0] == 3 && b.shape()[1] == 4);
  CHECK(b.strides()[1] == 0);
  CHECK(b.buffer().get() == a.buffer().get());
  CHECK(!b.writeable());
  CHECK_THROWS_AS(b.set_item<double>({0, 0}, 1.0), value_error);
  CHECK_THROWS_AS(broadcast_shapes({3}, {4}), value_error);
  CHECK(broadcast_shapes({3, 1}, {1, 4}) == Shape({3, 4}));
}

TEST_CASE("fill and item, read-only rejection") {
  ndarray a({2, 2}, kFloat64);
  a.fill<double>(3.5);
  CHECK(a.item<double>({1, 1}) == 3.5);
  ndarray ro = a.broadcast_to({2, 2});
  CHECK_THROWS_AS(ro.fill<double>(1.0), value_error);
}

TEST_CASE("out-of-range and axis errors") {
  ndarray a = iota({3});
  CHECK_THROWS_AS(a.item<double>({5}), index_error);
  CHECK_THROWS_AS(iota({2, 2}).transpose({0, 0}), value_error);
  CHECK_THROWS_AS(normalize_axis(3, 2), axis_error);
}

TEST_CASE("iterate transposed view in C order matches NumPy") {
  ndarray a = iota({2, 3});
  ndarray t = a.transpose();
  std::vector<double> got;
  t.for_each_offset([&](int64_t off) {
    double d; std::memcpy(&d, t.buffer()->data() + off, sizeof(double)); got.push_back(d);
  });
  // a.T.ravel('C') == [0,3,1,4,2,5]
  std::vector<double> want = {0, 3, 1, 4, 2, 5};
  CHECK(got == want);
}

TEST_CASE("reshape values vs NumPy oracle") {
  ndarray r = iota({2, 6}).reshape({3, 4});
  auto o = npt::oracle("a=np.arange(12.0).reshape(2,6).reshape(3,4)");
  if (!o) { std::fprintf(stderr, "  [skip] reshape oracle (no numpy)\n"); return; }
  CHECK(allclose(r, *o));
}
