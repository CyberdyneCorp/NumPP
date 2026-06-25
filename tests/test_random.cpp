#include <cstdint>
#include <vector>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
using numpp::random::Generator;
using numpp::random::PCG64;
using numpp::random::SeedSequence;

TEST_CASE("SeedSequence.generate_state matches numpy exactly") {
  SeedSequence ss(123);
  std::vector<uint32_t> got = ss.generate_state32(4);
  std::vector<uint32_t> want = {693650678u, 2973253328u, 387799215u, 2723177262u};
  CHECK(got == want);
}

TEST_CASE("PCG64 random_raw is bit-exact with numpy.random.PCG64(123)") {
  Generator g(123);
  ndarray raw = g.random_raw(5);
  std::vector<uint64_t> want = {12587170189557361101ull, 992822559630912803ull,
                                4064922177151560342ull, 3401059606364785669ull,
                                3244891138370822358ull};
  for (int i = 0; i < 5; ++i) CHECK(raw.item<uint64_t>({i}) == want[i]);
}

TEST_CASE("PCG64 raw matches numpy oracle (dynamic)") {
  Generator g(2024);
  ndarray raw = g.random_raw(8);
  auto o = npt::oracle("a=np.random.PCG64(2024).random_raw(8).astype(np.uint64)");
  if (!o) { std::fprintf(stderr, "  [skip] pcg raw oracle\n"); return; }
  for (int i = 0; i < 8; ++i) CHECK(raw.item<uint64_t>({i}) == o->item<uint64_t>({i}));
}

TEST_CASE("Generator.random matches numpy exactly") {
  Generator g(123);
  ndarray r = g.random(5);
  std::vector<double> want = {0.6823518632481435, 0.053821018802222675, 0.22035987277261138,
                              0.1843718106986697, 0.17590590108503035};
  for (int i = 0; i < 5; ++i) CHECK(std::abs(r.item<double>({i}) - want[i]) < 1e-15);
}

TEST_CASE("Generator.random vs oracle and shape") {
  Generator g(7);
  ndarray r = g.random({2, 3});
  CHECK(r.shape() == Shape({2, 3}));
  auto o = npt::oracle("a=np.random.Generator(np.random.PCG64(7)).random((2,3))");
  if (!o) { std::fprintf(stderr, "  [skip] random oracle\n"); return; }
  CHECK(allclose(r, *o, 1e-15, 1e-18));
}
