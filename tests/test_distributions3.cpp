#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::random;

static void hg_stats(const ndarray& a, double& mean, int64_t& lo, int64_t& hi) {
  const int64_t n = a.size();
  double s = 0;
  lo = a.item<int64_t>({0});
  hi = lo;
  for (int64_t i = 0; i < n; ++i) {
    int64_t v = a.item<int64_t>({i});
    s += v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  mean = s / n;
}

TEST_CASE("hypergeometric mean and range (20,10,5)") {
  random::Generator __g1(11);
  auto a = random::hypergeometric(__g1, 20, 10, 5, {200000});
  double mean; int64_t lo, hi;
  hg_stats(a, mean, lo, hi);
  const double theory = 5.0 * 20.0 / 30.0;
  CHECK(std::abs(mean - theory) < 0.02);
  CHECK(lo >= 0);
  CHECK(hi <= 5);  // min(nsample, ngood)
  CHECK(lo >= 0);  // max(0, nsample-nbad)=0
}

TEST_CASE("hypergeometric mean and range (5,15,8)") {
  random::Generator __g2(12);
  auto a = random::hypergeometric(__g2, 5, 15, 8, {200000});
  double mean; int64_t lo, hi;
  hg_stats(a, mean, lo, hi);
  const double theory = 8.0 * 5.0 / 20.0;
  CHECK(std::abs(mean - theory) < 0.02);
  CHECK(lo >= 0);          // max(0, 8-15)=0
  CHECK(hi <= 5);          // min(8,5)=5
}

TEST_CASE("hypergeometric forced lower bound (3,2,4)") {
  // nsample-nbad = 4-2 = 2 -> minimum count is 2
  random::Generator __g3(13);
  auto a = random::hypergeometric(__g3, 3, 2, 4, {50000});
  double mean; int64_t lo, hi;
  hg_stats(a, mean, lo, hi);
  CHECK(lo >= 2);
  CHECK(hi <= 3);
}

TEST_CASE("hypergeometric shape preserved") {
  random::Generator __g4(14);
  auto a = random::hypergeometric(__g4, 10, 10, 5, {3, 4});
  CHECK((a.shape() == Shape{3, 4}));
  CHECK(a.dtype() == kInt64);
}

TEST_CASE("noncentral_f mean (df1=5,df2=10,nonc=3)") {
  random::Generator __g5(21);
  auto a = random::noncentral_f(__g5, 5.0, 10.0, 3.0, {200000});
  double m, v;
  const int64_t n = a.size();
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += a.item<double>({i});
  m = s / n;
  (void)v;
  const double theory = 10.0 * (5.0 + 3.0) / (5.0 * (10.0 - 2.0));
  CHECK(std::abs(m - theory) / theory < 0.05);
}

TEST_CASE("noncentral_f mean (df1=8,df2=20,nonc=5)") {
  random::Generator __g6(22);
  auto a = random::noncentral_f(__g6, 8.0, 20.0, 5.0, {200000});
  const int64_t n = a.size();
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += a.item<double>({i});
  const double m = s / n;
  const double theory = 20.0 * (8.0 + 5.0) / (8.0 * (20.0 - 2.0));
  CHECK(std::abs(m - theory) / theory < 0.05);
  CHECK(a.dtype() == kFloat64);
}

TEST_CASE("noncentral_f all positive and shaped") {
  random::Generator __g7(23);
  auto a = random::noncentral_f(__g7, 4.0, 12.0, 2.0, {2, 5});
  CHECK((a.shape() == Shape{2, 5}));
  ndarray flat = a.ascontiguousarray().reshape({a.size()});
  for (int64_t i = 0; i < flat.size(); ++i) {
    CHECK((flat.item<double>({i}) > 0.0));
  }
}

TEST_CASE("bytes length, range, and determinism") {
  random::Generator __g8(31);
  auto a = random::bytes(__g8, 100);
  CHECK(a.size() == 100);
  CHECK(a.dtype() == kUInt8);
  for (int64_t i = 0; i < a.size(); ++i) {
    int v = a.item<uint8_t>({i});
    CHECK(v >= 0);
    CHECK(v <= 255);
  }
  // determinism: same seed -> same bytes
  random::Generator __g9(31);
  auto b = random::bytes(__g9, 100);
  bool same = true;
  for (int64_t i = 0; i < a.size(); ++i)
    if (a.item<uint8_t>({i}) != b.item<uint8_t>({i})) same = false;
  CHECK(same);
  // non-multiple-of-8 length still exact
  random::Generator __g10(32);
  auto c = random::bytes(__g10, 13);
  CHECK(c.size() == 13);
}
