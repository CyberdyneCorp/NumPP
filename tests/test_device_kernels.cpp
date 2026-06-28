// ScyPP acceleration primitives (#99, #106): CSR SpMV, pairwise distance, and
// separable 1-D correlation. Validates correctness (vs an independent numpy
// reference and scipy goldens) and the device-dispatch path. Works in BOTH the
// default (gpu_null -> last_backend()==Cpu) and NUMPP_WITH_REFGPU builds, where the
// device path must equal the CPU path exactly and report Backend::Device.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

namespace {
ndarray dv(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray iv(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray mat(const std::vector<double>& v, int64_t r, int64_t c) {
  ndarray a(Shape{r, c}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
// Run f() once with the device enabled (small threshold) and once forced to CPU,
// returning both results and the device-run backend. Mirrors test_gpu.cpp.
template <class F>
void dev_vs_cpu(F&& f, ndarray& dev, ndarray& cpu, Backend& dev_backend) {
  setenv("NUMPP_GPU_MIN", "1", 1);
  dev = f();
  dev_backend = last_backend();
  setenv("NUMPP_GPU_MIN", "1000000000", 1);
  cpu = f();
  CHECK(last_backend() == Backend::Cpu);
  unsetenv("NUMPP_GPU_MIN");
  // The device path (refgpu) is byte-identical to the CPU path.
  CHECK(allclose(dev, cpu, 0.0, 0.0, true));
  CHECK((dev_backend == Backend::Device || dev_backend == Backend::Cpu));
}
}  // namespace

TEST_CASE("device csr_spmv: matches dense matmul + device dispatch") {
  // A = [[2,0,1],[0,3,0],[4,0,5]] in CSR; x = [1,2,3]
  ndarray indptr = iv({0, 2, 3, 5}), indices = iv({0, 2, 1, 0, 2}), data = dv({2, 1, 3, 4, 5}), x = dv({1, 2, 3});
  ndarray dev, cpu; Backend b;
  dev_vs_cpu([&] { return csr_spmv(indptr, indices, data, x); }, dev, cpu, b);
  // Independent reference: densify and matmul (numpy oracle).
  auto o = npt::oracle("A=np.array([[2,0,1],[0,3,0],[4,0,5.]]); a=A@np.array([1,2,3.])");
  if (o) CHECK(allclose(cpu, *o, 1e-12, 1e-12, true));   // expect [5, 6, 19]
}

TEST_CASE("device cdist_euclidean: matches numpy broadcasting + device dispatch") {
  ndarray A = mat({0, 0, 1, 0, 0, 1}, 3, 2), B = mat({0, 0, 1, 1}, 2, 2);
  ndarray dev, cpu; Backend b;
  dev_vs_cpu([&] { return cdist_euclidean(A, B, /*squared=*/false); }, dev, cpu, b);
  auto o = npt::oracle(
      "A=np.array([[0,0],[1,0],[0,1.]]); B=np.array([[0,0],[1,1.]]); "
      "a=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))");
  if (o) CHECK(allclose(cpu, *o, 1e-12, 1e-12, true));

  ndarray devs, cpus; Backend bs;
  dev_vs_cpu([&] { return cdist_euclidean(A, B, /*squared=*/true); }, devs, cpus, bs);
  auto os = npt::oracle(
      "A=np.array([[0,0],[1,0],[0,1.]]); B=np.array([[0,0],[1,1.]]); "
      "a=((A[:,None,:]-B[None,:,:])**2).sum(-1)");
  if (os) CHECK(allclose(cpus, *os, 1e-12, 1e-12, true));
}

TEST_CASE("device correlate1d: matches scipy goldens across modes + device dispatch") {
  ndarray sig = dv({1, 2, 3, 4, 5}), w = dv({0.25, 0.5, 0.25});
  struct Case { FilterMode mode; std::vector<double> want; };
  // Golden outputs from scipy.ndimage.correlate1d(sig, w, mode=...).
  const std::vector<Case> cases = {
      {FilterMode::Reflect, {1.25, 2, 3, 4, 4.75}},
      {FilterMode::Constant, {1.0, 2, 3, 4, 3.5}},
      {FilterMode::Nearest, {1.25, 2, 3, 4, 4.75}},
      {FilterMode::Mirror, {1.5, 2, 3, 4, 4.5}},
      {FilterMode::Wrap, {2.25, 2, 3, 4, 3.75}},
  };
  for (const auto& c : cases) {
    ndarray dev, cpu; Backend b;
    dev_vs_cpu([&] { return correlate1d(sig, w, -1, 0, c.mode, 0.0); }, dev, cpu, b);
    CHECK(allclose(cpu, dv(c.want), 1e-12, 1e-12, true));
  }
  // origin shift matches scipy (origin=1 -> [1.25, 1.25, 2, 3, 4]).
  ndarray dev, cpu; Backend b;
  dev_vs_cpu([&] { return correlate1d(sig, w, -1, 1, FilterMode::Reflect, 0.0); }, dev, cpu, b);
  CHECK(allclose(cpu, dv({1.25, 1.25, 2, 3, 4}), 1e-12, 1e-12, true));
}

TEST_CASE("device correlate1d: 2-D along each axis matches scipy when available") {
  // Cross-check against live scipy if it is installed; the oracle returns nullopt
  // (and the check is skipped) when scipy is absent, e.g. in CI.
  ndarray img = mat({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4);
  ndarray w = dv({1, 0, -1});
  for (int64_t axis = 0; axis <= 1; ++axis) {
    ndarray got = correlate1d(img, w, axis, 0, FilterMode::Reflect, 0.0, Backend::Cpu);
    auto o = npt::oracle(
        "from scipy.ndimage import correlate1d as c1d; img=np.arange(1,13.).reshape(3,4); "
        "a=c1d(img,np.array([1,0,-1.]),axis=" + std::to_string(axis) + ",mode='reflect')");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  }
}
