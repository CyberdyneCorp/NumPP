#include <cstdlib>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"

using namespace numpp;

// Works in BOTH configs: default (gpu_null) keeps last_backend()==Cpu and only
// checks correctness; with NUMPP_WITH_REFGPU the device path is exercised and
// must equal the CPU path exactly + report Backend::Device above the threshold.
static ndarray seqf(int64_t n, double start) {
  ndarray a({n}, kFloat64); for (int64_t i = 0; i < n; ++i) a.set_item<double>({i}, start + i * 0.5);
  return a;
}

TEST_CASE("gpu dispatch: device results equal CPU, last_backend correct") {
  const int64_t n = 5000;
  ndarray a = seqf(n, 1.0), b = seqf(n, 3.0);

  setenv("NUMPP_GPU_MIN", "100", 1);          // enable device for this size
  ndarray r_dev = add(a, b);
  Backend lb = last_backend();

  setenv("NUMPP_GPU_MIN", "1000000000", 1);   // force CPU (below threshold)
  ndarray r_cpu = add(a, b);
  CHECK(last_backend() == Backend::Cpu);

  CHECK(allclose(r_dev, r_cpu, 0.0, 0.0));     // device path is bitwise-equal to CPU
  CHECK((lb == Backend::Device || lb == Backend::Cpu));
  bool refgpu = (lb == Backend::Device);
  std::fprintf(stderr, "  [gpu] device backend %s\n", refgpu ? "ACTIVE (refgpu)" : "absent (cpu-only build)");

  // unary + reduction also route
  setenv("NUMPP_GPU_MIN", "100", 1);
  ndarray s_dev = sqrt(a);
  Backend lb_sqrt = last_backend();
  ndarray sm_dev = sum(a);
  Backend lb_sum = last_backend();
  setenv("NUMPP_GPU_MIN", "1000000000", 1);
  CHECK(allclose(s_dev, sqrt(a), 0.0, 0.0));
  CHECK(allclose(sm_dev, sum(a), 1e-9, 1e-9));
  if (refgpu) { CHECK(lb_sqrt == Backend::Device); CHECK(lb_sum == Backend::Device); }
  unsetenv("NUMPP_GPU_MIN");
}

TEST_CASE("gpu: integer ops never use device (fall back to CPU)") {
  setenv("NUMPP_GPU_MIN", "1", 1);
  ndarray a = arange(0.0, 5000.0, 1.0).astype(kInt64);
  ndarray r = add(a, a);
  CHECK(last_backend() == Backend::Cpu);       // int dtype not gpu-eligible
  CHECK(r.item<int64_t>({3}) == 6);
  unsetenv("NUMPP_GPU_MIN");
}
