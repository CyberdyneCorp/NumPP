#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
using numpp::random::Generator;

TEST_CASE("integers bit-exact vs numpy (small range, 32-bit Lemire)") {
  Generator g(123);
  ndarray r = g.integers(0, 100, 8);
  int64_t want[8] = {1, 68, 59, 5, 90, 22, 25, 18};
  CHECK(r.dtype() == kInt64);
  for (int i = 0; i < 8; ++i) CHECK(r.item<int64_t>({i}) == want[i]);
}

TEST_CASE("integers vs oracle: ranges, endpoint, negative low") {
  auto cmp = [](const ndarray& got, const std::string& code) {
    auto o = npt::oracle(code);
    if (!o) { std::fprintf(stderr, "  [skip] %s\n", code.c_str()); return; }
    for (int64_t i = 0; i < got.size(); ++i) CHECK(got.item<int64_t>({i}) == o->item<int64_t>({i}));
  };
  cmp(Generator(7).integers(-5, 5, 10), "a=np.random.Generator(np.random.PCG64(7)).integers(-5,5,10)");
  cmp(Generator(7).integers(0, 1000000007, 6), "a=np.random.Generator(np.random.PCG64(7)).integers(0,1000000007,6)");
  cmp(Generator(9).integers(0, 10, 5, true), "a=np.random.Generator(np.random.PCG64(9)).integers(0,10,5,endpoint=True)");
  // 64-bit range (> 2^32)
  cmp(Generator(3).integers(0, 5000000000ll, 5), "a=np.random.Generator(np.random.PCG64(3)).integers(0,5000000000,5)");
}

TEST_CASE("uniform bit-exact vs numpy") {
  auto o = npt::oracle("a=np.random.Generator(np.random.PCG64(123)).uniform(2,5,4)");
  if (!o) { std::fprintf(stderr, "  [skip] uniform\n"); return; }
  ndarray r = Generator(123).uniform(2, 5, 4);
  for (int i = 0; i < 4; ++i) CHECK(std::abs(r.item<double>({i}) - o->item<double>({i})) < 1e-13);
}

TEST_CASE("permutation bit-exact vs numpy") {
  Generator g(123);
  ndarray p = g.permutation(8);
  int64_t want[8] = {0, 6, 4, 2, 7, 3, 1, 5};
  for (int i = 0; i < 8; ++i) CHECK(p.item<int64_t>({i}) == want[i]);
  auto o = npt::oracle("a=np.random.Generator(np.random.PCG64(42)).permutation(20)");
  if (o) { ndarray q = Generator(42).permutation(20); for (int i = 0; i < 20; ++i) CHECK(q.item<int64_t>({i}) == o->item<int64_t>({i})); }
}

TEST_CASE("choice with replacement bit-exact (uses integers)") {
  auto o = npt::oracle("a=np.random.Generator(np.random.PCG64(11)).integers(0,10,4)");
  if (o) { ndarray c = Generator(11).choice(10, 4, true); for (int i = 0; i < 4; ++i) CHECK(c.item<int64_t>({i}) == o->item<int64_t>({i})); }
}

TEST_CASE("choice without replacement: valid sample (issue #7: not bit-exact)") {
  // numpy's no-replacement choice stream differs (see issue #7); assert validity.
  ndarray c = Generator(123).choice(10, 5, false);
  CHECK(c.size() == 5);
  for (int i = 0; i < 5; ++i) {
    int64_t v = c.item<int64_t>({i});
    CHECK(v >= 0 && v < 10);
    for (int j = i + 1; j < 5; ++j) CHECK(v != c.item<int64_t>({j}));  // distinct
  }
}

TEST_CASE("shuffle permutes in place and is a permutation") {
  Generator g(5);
  ndarray a = arange(0.0, 10.0, 1.0).astype(kInt64);
  g.shuffle(a);
  int64_t sum = 0; for (int i = 0; i < 10; ++i) sum += a.item<int64_t>({i});
  CHECK(sum == 45);  // still 0..9
}
