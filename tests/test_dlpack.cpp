#include "numpp/numpp.hpp"
#include "numpp/interop/interop.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <filesystem>
#include <string>

using namespace numpp;
using namespace numpp::interop;

using namespace numpp;
using namespace numpp::interop;

TEST_CASE("dlpack round-trip float64 2x3") {
    ndarray x(Shape{2, 3}, kFloat64, Order::C);
    for (int64_t i = 0; i < 6; ++i) x.set_item<double>({i / 3, i % 3}, static_cast<double>(i) + 0.5);

    DLManagedTensor* t = to_dlpack(x);
    CHECK(t->dl_tensor.ndim == 2);
    CHECK(t->dl_tensor.dtype.code == kDLFloat);
    CHECK(t->dl_tensor.dtype.bits == 64);
    CHECK(t->dl_tensor.dtype.lanes == 1);
    CHECK(t->dl_tensor.device.device_type == kDLCPU);
    CHECK(t->dl_tensor.shape[0] == 2);
    CHECK(t->dl_tensor.shape[1] == 3);
    CHECK(t->dl_tensor.strides[0] == 3);  // element strides
    CHECK(t->dl_tensor.strides[1] == 1);

    ndarray y = from_dlpack(t);
    CHECK(y.shape() == x.shape());
    CHECK(y.dtype() == x.dtype());
    CHECK(allclose(y, x, 0, 0, true));
}

TEST_CASE("dlpack round-trip int64") {
    ndarray x(Shape{4}, kInt64, Order::C);
    for (int64_t i = 0; i < 4; ++i) x.set_item<int64_t>({i}, i * 7 - 3);

    DLManagedTensor* t = to_dlpack(x);
    CHECK(t->dl_tensor.ndim == 1);
    CHECK(t->dl_tensor.dtype.code == kDLInt);
    CHECK(t->dl_tensor.dtype.bits == 64);

    ndarray y = from_dlpack(t);
    CHECK(y.shape() == x.shape());
    CHECK(allclose(y, x, 0, 0, true));
}

TEST_CASE("dlpack handles non-contiguous (transposed) input") {
  ndarray x(Shape{2, 3}, kFloat64, Order::C);
  for (int64_t i = 0; i < 6; ++i) x.set_item<double>({i / 3, i % 3}, static_cast<double>(i));
  ndarray xt = x.transpose();
  DLManagedTensor* t = to_dlpack(xt);
  CHECK(t->dl_tensor.ndim == 2);
  CHECK(t->dl_tensor.shape[0] == 3);
  CHECK(t->dl_tensor.shape[1] == 2);
  CHECK(t->dl_tensor.strides[0] == 2);
  CHECK(t->dl_tensor.strides[1] == 1);
  ndarray y = from_dlpack(t);
  CHECK(allclose(y, xt, 0, 0, true));
}
