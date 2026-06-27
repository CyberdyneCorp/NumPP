#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <complex>
#include <optional>
#include <vector>

using namespace numpp;

// Oracle snippets prefer the Array API alias name and fall back to the
// equivalent classic NumPy call when the installed NumPy lacks the alias.

TEST_CASE("matrix_transpose 3d real vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(24.).reshape(2, 3, 4)
try:
    a = np.matrix_transpose(x)
except AttributeError:
    a = np.swapaxes(x, -1, -2)
)");
  if (!o) return;
  auto x = arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4});
  auto r = linalg::matrix_transpose(x);
  CHECK((r.shape() == Shape{2, 4, 3}));
  CHECK((allclose(r, *o, 1e-9, 1e-12, true)));
}

TEST_CASE("matrix_transpose requires 2-D") {
  auto x = arange(0.0, 4.0, 1.0, kFloat64);
  CHECK_THROWS_AS(linalg::matrix_transpose(x), value_error);
}

TEST_CASE("permute_dims vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(24.).reshape(2, 3, 4)
try:
    a = np.permute_dims(x, (2, 0, 1))
except AttributeError:
    a = np.transpose(x, (2, 0, 1))
)");
  if (!o) return;
  auto x = arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4});
  auto r = linalg::permute_dims(x, {2, 0, 1});
  CHECK((r.shape() == Shape{4, 2, 3}));
  CHECK((allclose(r, *o, 1e-9, 1e-12, true)));
}

TEST_CASE("vecdot real vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(12.).reshape(3, 4)
y = np.arange(12.).reshape(3, 4) + 1.0
try:
    a = np.vecdot(x, y)
except AttributeError:
    a = np.sum(np.conj(x) * y, axis=-1)
)");
  if (!o) return;
  auto x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
  auto y = add(arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4}),
               scalar_like(1.0, kFloat64, true));
  auto r = linalg::vecdot(x, y);
  CHECK((r.shape() == Shape{3}));
  CHECK((allclose(r, *o, 1e-9, 1e-12, true)));
}

TEST_CASE("vecdot complex conjugates first arg vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
y = np.array([[2-1j, 1+1j], [1-1j, 3+2j]])
try:
    a = np.vecdot(x, y)
except AttributeError:
    a = np.sum(np.conj(x) * y, axis=-1)
)");
  if (!o) return;
  using cd = std::complex<double>;
  ndarray x(Shape{2, 2}, kComplex128);
  x.set_item<cd>({0, 0}, cd(1, 2));
  x.set_item<cd>({0, 1}, cd(3, -1));
  x.set_item<cd>({1, 0}, cd(0, 1));
  x.set_item<cd>({1, 1}, cd(2, 0));
  ndarray y(Shape{2, 2}, kComplex128);
  y.set_item<cd>({0, 0}, cd(2, -1));
  y.set_item<cd>({0, 1}, cd(1, 1));
  y.set_item<cd>({1, 0}, cd(1, -1));
  y.set_item<cd>({1, 1}, cd(3, 2));
  auto r = linalg::vecdot(x, y);
  CHECK((r.shape() == Shape{2}));
  CHECK((allclose(r, *o, 1e-9, 1e-12, true)));
}

TEST_CASE("vector_norm default 2-norm vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(1., 7.).reshape(2, 3)
try:
    a = np.linalg.vector_norm(x)
except AttributeError:
    a = np.linalg.norm(x.ravel())
)");
  if (!o) return;
  auto x = arange(1.0, 7.0, 1.0, kFloat64).reshape({2, 3});
  CHECK((allclose(linalg::vector_norm(x), *o, 1e-9, 1e-12, true)));
}

TEST_CASE("vector_norm ord=1 over axis vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.array([[1., -2., 3.], [-4., 5., -6.]])
try:
    a = np.linalg.vector_norm(x, axis=1, ord=1)
except AttributeError:
    a = np.linalg.norm(x, axis=1, ord=1)
)");
  if (!o) return;
  ndarray x(Shape{2, 3}, kFloat64);
  x.set_item<double>({0, 0}, 1.0);  x.set_item<double>({0, 1}, -2.0); x.set_item<double>({0, 2}, 3.0);
  x.set_item<double>({1, 0}, -4.0); x.set_item<double>({1, 1}, 5.0);  x.set_item<double>({1, 2}, -6.0);
  auto r = linalg::vector_norm(x, 1, 1.0);
  CHECK((r.shape() == Shape{2}));
  CHECK((allclose(r, *o, 1e-9, 1e-12, true)));
}

TEST_CASE("vector_norm ord=3 flattened vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(1., 7.)
try:
    a = np.linalg.vector_norm(x, ord=3)
except AttributeError:
    a = np.linalg.norm(x, ord=3)
)");
  if (!o) return;
  auto x = arange(1.0, 7.0, 1.0, kFloat64);
  CHECK((allclose(linalg::vector_norm(x, std::nullopt, 3.0), *o, 1e-9, 1e-12, true)));
}

TEST_CASE("matrix_norm Frobenius vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(R"(
x = np.arange(1., 7.).reshape(2, 3)
try:
    a = np.linalg.matrix_norm(x)
except AttributeError:
    a = np.linalg.norm(x)
)");
  if (!o) return;
  auto x = arange(1.0, 7.0, 1.0, kFloat64).reshape({2, 3});
  CHECK((allclose(linalg::matrix_norm(x), *o, 1e-9, 1e-12, true)));
}
