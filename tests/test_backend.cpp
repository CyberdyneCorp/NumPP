#include <cstdlib>

#include "numpp/numpp.hpp"
#include "numpp/backend/config.hpp"  // NUMPP_WITH_* feature macros for the #if guards
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static ndarray seq(const Shape& s, double start, DType dt) {
  ndarray a(s, dt, Order::C);
  double v = start;
  a.for_each_offset([&](int64_t off) {
    visit_dtype(dt.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      T x;
      if constexpr (std::is_same_v<T, std::complex<double>>) x = std::complex<double>(v, 0);
      else x = static_cast<T>(v);
      std::memcpy(a.buffer()->data() + off, &x, sizeof(T));
    });
    v += 1;
  });
  return a;
}

TEST_CASE("capability registry on CPU-only build") {
  const auto& cap = CapabilityRegistry::instance();
  // Default build: all backend flags OFF -> nothing available.
#if !NUMPP_WITH_BLAS
  CHECK(!cap.has_blas());
#endif
#if !NUMPP_WITH_METAL
  CHECK(!cap.gpu_available(Backend::Metal));
#endif
#if !NUMPP_WITH_CUDA
  CHECK(!cap.gpu_available(Backend::Cuda));
#endif
#if !NUMPP_WITH_VULKAN
  CHECK(!cap.gpu_available(Backend::Vulkan));
#endif
#if !NUMPP_WITH_OPENCL
  CHECK(!cap.gpu_available(Backend::OpenCL));
#endif
}

TEST_CASE("matmul correctness and CPU dispatch") {
  ndarray a = seq({2, 3}, 1, kFloat64);
  ndarray b = seq({3, 2}, 1, kFloat64);
  ndarray c = matmul(a, b, Backend::Cpu);
  CHECK(last_backend() == Backend::Cpu);
  CHECK(c.shape()[0] == 2 && c.shape()[1] == 2);
  // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
  CHECK(c.item<double>({0, 0}) == 22.0);
  CHECK(c.item<double>({1, 1}) == 64.0);
}

TEST_CASE("matmul vs NumPy oracle") {
  ndarray a = seq({4, 5}, 1, kFloat64);
  ndarray b = seq({5, 3}, 2, kFloat64);
  ndarray c = matmul(a, b);
  auto o = npt::oracle(
      "a=np.arange(1,21,dtype=np.float64).reshape(4,5)@np.arange(2,17,dtype=np.float64).reshape(5,3)");
  if (!o) { std::fprintf(stderr, "  [skip] matmul oracle (no numpy)\n"); return; }
  CHECK(allclose(c, *o));
}

TEST_CASE("force-CPU via env target") {
  setenv("NUMPP_GPU_TARGET", "cpu", 1);
  ndarray a = seq({2, 2}, 1, kFloat64);
  matmul(a, a);
  CHECK(last_backend() == Backend::Cpu);
  unsetenv("NUMPP_GPU_TARGET");
}

TEST_CASE("explicit unavailable backend errors") {
  ndarray a = seq({2, 2}, 1, kFloat64);
#if !NUMPP_WITH_BLAS
  CHECK_THROWS_AS(matmul(a, a, Backend::Blas), not_implemented_error);
#endif
#if !NUMPP_WITH_CUDA
  setenv("NUMPP_GPU_TARGET", "cuda", 1);
  CHECK_THROWS_AS(matmul(a, a), not_implemented_error);  // explicit GPU, not compiled
  unsetenv("NUMPP_GPU_TARGET");
#endif
  // auto (unset) degrades silently to CPU.
  matmul(a, a);
  CHECK(last_backend() == Backend::Cpu);
}

TEST_CASE("backend equivalence: CPU vs BLAS") {
  const auto& cap = CapabilityRegistry::instance();
  if (!cap.has_blas()) { std::fprintf(stderr, "  [skip] CPU/BLAS equivalence (no BLAS)\n"); return; }
  ndarray a = seq({8, 8}, 1, kFloat64);
  ndarray b = seq({8, 8}, 3, kFloat64);
  ndarray cpu = matmul(a, b, Backend::Cpu);
  ndarray blas = matmul(a, b, Backend::Blas);
  CHECK(allclose(cpu, blas));
}
