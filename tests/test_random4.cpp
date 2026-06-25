#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
using numpp::random::MT19937;
using numpp::random::RandomState;

TEST_CASE("RandomState.random_sample bit-exact with numpy") {
  RandomState rs(123);
  ndarray r = rs.random_sample(3);
  double want[3] = {0.6964691855978616, 0.28613933495037946, 0.2268514535642031};
  for (int i = 0; i < 3; ++i) CHECK(std::abs(r.item<double>({i}) - want[i]) < 1e-15);
}

TEST_CASE("RandomState.randint bit-exact with numpy") {
  RandomState rs(123);
  ndarray r = rs.randint(0, 100, 5);
  int64_t want[5] = {66, 92, 98, 17, 83};
  for (int i = 0; i < 5; ++i) CHECK(r.item<int64_t>({i}) == want[i]);
}

TEST_CASE("RandomState.randn bit-exact with numpy (legacy gauss)") {
  RandomState rs(123);
  ndarray r = rs.randn(3);
  double want[3] = {-1.0856306033005612, 0.9973454465835858, 0.28297849805199204};
  for (int i = 0; i < 3; ++i) CHECK(std::abs(r.item<double>({i}) - want[i]) < 1e-14);
}

TEST_CASE("RandomState vs oracle: random/randint/normal/uniform") {
  auto chkd = [](const ndarray& got, const std::string& code) {
    auto o = npt::oracle(code);
    if (!o) { std::fprintf(stderr, "  [skip] %s\n", code.c_str()); return; }
    for (int64_t i = 0; i < got.size(); ++i) CHECK(std::abs(got.item<double>({i}) - o->item<double>({i})) < 1e-12);
  };
  auto chki = [](const ndarray& got, const std::string& code) {
    auto o = npt::oracle(code);
    if (!o) { std::fprintf(stderr, "  [skip] %s\n", code.c_str()); return; }
    for (int64_t i = 0; i < got.size(); ++i) CHECK(got.item<int64_t>({i}) == o->item<int64_t>({i}));
  };
  chkd(RandomState(7).random_sample(6), "a=np.random.RandomState(7).random_sample(6)");
  chki(RandomState(7).randint(0, 1000, 6), "a=np.random.RandomState(7).randint(0,1000,6)");
  chkd(RandomState(7).normal(3.0, 2.0, 5), "a=np.random.RandomState(7).normal(3,2,5)");
  chkd(RandomState(7).uniform(-1.0, 1.0, 5), "a=np.random.RandomState(7).uniform(-1,1,5)");
}

TEST_CASE("MT19937 legacy seeding matches RandomState bit stream") {
  // MT19937(seed).next32 == raw stream behind RandomState(seed)
  MT19937 mt(123);
  auto o = npt::oracle("a=np.random.RandomState(123)._bit_generator.random_raw(4).astype(np.uint64)");
  if (!o) { std::fprintf(stderr, "  [skip] mt raw\n"); return; }
  for (int i = 0; i < 4; ++i) CHECK(static_cast<uint64_t>(mt.next32()) == o->item<uint64_t>({i}));
}
