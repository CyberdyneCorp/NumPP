#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::random;

TEST_CASE("Philox random_raw bit-exact vs numpy") {
  using numpp::random::Philox;
  for (uint64_t seed : {12345ull, 0ull, 987654321ull}) {
    Philox p(seed);
    const int64_t n = 8;
    ndarray r = p.random_raw(n);
    auto o = npt::oracle("a=np.random.Philox(" + std::to_string(seed) +
                         ").random_raw(" + std::to_string(n) + ")");
    if (o) {
      bool ok = true;
      for (int64_t i = 0; i < n; ++i)
        ok = ok && (r.item<uint64_t>({i}) == o->item<uint64_t>({i}));
      CHECK(ok);
    }
  }
}

TEST_CASE("MT19937 random_raw bit-exact vs numpy") {
  using numpp::random::MT19937BitGen;
  for (uint64_t seed : {12345ull, 0ull, 42ull, 999999ull}) {
    MT19937BitGen g(seed);
    const int64_t n = 10;
    ndarray r = g.random_raw(n);
    auto o = npt::oracle("a=np.random.MT19937(" + std::to_string(seed) +
                         ").random_raw(" + std::to_string(n) + ")");
    if (o) {
      bool ok = true;
      for (int64_t i = 0; i < n; ++i)
        ok = ok && (r.item<uint64_t>({i}) == o->item<uint64_t>({i}));
      CHECK(ok);
    }
  }
}

TEST_CASE("Philox next64 streams sequentially across buffer boundary") {
  using numpp::random::Philox;
  Philox p(12345);
  ndarray bulk = Philox(12345).random_raw(10);
  bool ok = true;
  for (int64_t i = 0; i < 10; ++i) ok = ok && (p.next64() == bulk.item<uint64_t>({i}));
  CHECK(ok);
}

TEST_CASE("SFC64 random_raw bit-exact vs numpy") {
  using numpp::random::SFC64;
  for (uint64_t seed : {999ull, 1ull, 424242ull}) {
    SFC64 g(seed);
    const int64_t n = 6;
    ndarray r = g.random_raw(n);
    auto o = npt::oracle("a=np.random.SFC64(" + std::to_string(seed) +
                         ").random_raw(" + std::to_string(n) + ")");
    if (o) {
      bool ok = true;
      for (int64_t i = 0; i < n; ++i)
        ok = ok && (r.item<uint64_t>({i}) == o->item<uint64_t>({i}));
      CHECK(ok);
    }
  }
}

TEST_CASE("SFC64 next64 matches bulk random_raw") {
  using numpp::random::SFC64;
  SFC64 g(999);
  ndarray bulk = SFC64(999).random_raw(8);
  bool ok = true;
  for (int64_t i = 0; i < 8; ++i) ok = ok && (g.next64() == bulk.item<uint64_t>({i}));
  CHECK(ok);
}
