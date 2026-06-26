#include "numpp/random/bitgenerators.hpp"

#include "numpp/random/bitgen.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <array>
#include <cstdint>
#include <vector>


namespace numpp {

namespace random {

namespace {

// constants for Philox4x64-10
constexpr uint64_t PHILOX_M0 = 0xD2B74407B1CE6E93ULL;
constexpr uint64_t PHILOX_M1 = 0xCA5A826395121157ULL;
constexpr uint64_t PHILOX_W0 = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t PHILOX_W1 = 0xBB67AE8584CAA73BULL;

inline void mulhilo(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
  u128 p = static_cast<u128>(a) * b;
  hi = static_cast<uint64_t>(p >> 64);
  lo = static_cast<uint64_t>(p);
}

inline std::array<uint64_t, 4> philox_round(const std::array<uint64_t, 4>& ctr,
                                            const std::array<uint64_t, 2>& key) {
  uint64_t hi0, lo0, hi1, lo1;
  mulhilo(PHILOX_M0, ctr[0], hi0, lo0);
  mulhilo(PHILOX_M1, ctr[2], hi1, lo1);
  std::array<uint64_t, 4> out;
  out[0] = hi1 ^ ctr[1] ^ key[0];
  out[1] = lo1;
  out[2] = hi0 ^ ctr[3] ^ key[1];
  out[3] = lo0;
  return out;
}

inline std::array<uint64_t, 4> philox10(const std::array<uint64_t, 4>& ctr,
                                        const std::array<uint64_t, 2>& key) {
  std::array<uint64_t, 4> ct = ctr;
  std::array<uint64_t, 2> k = key;
  for (int r = 0; r < 10; ++r) {
    ct = philox_round(ct, k);
    if (r < 9) {
      k[0] += PHILOX_W0;
      k[1] += PHILOX_W1;
    }
  }
  return ct;
}

}  // namespace

Philox::Philox(uint64_t seed) {
  std::vector<uint64_t> s = SeedSequence(seed).generate_state64(2);
  key_ = {s[0], s[1]};
  counter_ = {0, 0, 0, 0};
  buffer_ = {0, 0, 0, 0};
  buffer_pos_ = 4;
}

uint64_t Philox::next64() {
  if (buffer_pos_ == 4) {
    buffer_ = philox10(counter_, key_);
    // increment 256-bit counter
    if (++counter_[0] == 0)
      if (++counter_[1] == 0)
        if (++counter_[2] == 0) ++counter_[3];
    buffer_pos_ = 0;
  }
  return buffer_[buffer_pos_++];
}

ndarray Philox::random_raw(int64_t n) {
  ndarray out(Shape{n}, kUInt64, Order::C);
  uint64_t* p = out.typed_data<uint64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = next64();
  return out;
}

SFC64::SFC64(uint64_t seed) {
  std::vector<uint64_t> s = SeedSequence(seed).generate_state64(3);
  a_ = s[0];
  b_ = s[1];
  c_ = s[2];
  counter_ = 1;
  for (int i = 0; i < 12; ++i) next64();  // numpy SFC64 uses 12 warmup rounds
}

uint64_t SFC64::next64() {
  uint64_t tmp = a_ + b_ + counter_++;
  a_ = b_ ^ (b_ >> 11);
  b_ = c_ + (c_ << 3);
  c_ = ((c_ << 24) | (c_ >> 40)) + tmp;
  return tmp;
}

ndarray SFC64::random_raw(int64_t n) {
  ndarray out(Shape{n}, kUInt64, Order::C);
  uint64_t* p = out.typed_data<uint64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = next64();
  return out;
}

}  // namespace random

}  // namespace numpp
