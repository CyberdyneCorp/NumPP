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

// constants for Philox4x64-10. NOTE: numpy uses 0xD2E7470EE14C6C93 for M0 (its
// own historical constant, NOT the canonical Random123 0xD2B74407B1CE6E93) — this
// is required for bit-exact parity with numpy.random.Philox.
constexpr uint64_t PHILOX_M0 = 0xD2E7470EE14C6C93ULL;
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

// MT19937 tempering and twist (32-bit).
constexpr uint32_t MT_UPPER = 0x80000000u, MT_LOWER = 0x7fffffffu, MT_MATRIX = 0x9908b0dfu;

inline uint32_t mt_temper(uint32_t y) {
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= y >> 18;
  return y;
}

}  // namespace

MT19937BitGen::MT19937BitGen(uint64_t seed) {
  // numpy seeds MT19937 by filling the 624-word state from SeedSequence, forcing
  // the first word to 0x80000000, and starting at pos 623 (so the first draw
  // emits mt[623] before the first twist).
  std::vector<uint32_t> s = SeedSequence(seed).generate_state32(624);
  for (int i = 0; i < 624; ++i) mt_[i] = s[i];
  mt_[0] = 0x80000000u;
  pos_ = 623;
}

void MT19937BitGen::twist() {
  for (int i = 0; i < 624; ++i) {
    uint32_t y = (mt_[i] & MT_UPPER) | (mt_[(i + 1) % 624] & MT_LOWER);
    mt_[i] = mt_[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX : 0u);
  }
}

uint32_t MT19937BitGen::next32() {
  if (pos_ >= 624) { twist(); pos_ = 0; }
  return mt_temper(mt_[pos_++]);
}

ndarray MT19937BitGen::random_raw(int64_t n) {
  ndarray out(Shape{n}, kUInt64, Order::C);
  uint64_t* p = out.typed_data<uint64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = next32();
  return out;
}

Philox::Philox(uint64_t seed) {
  std::vector<uint64_t> s = SeedSequence(seed).generate_state64(2);
  key_ = {s[0], s[1]};
  counter_ = {0, 0, 0, 0};
  buffer_ = {0, 0, 0, 0};
  buffer_pos_ = 4;
}

uint64_t Philox::next64() {
  if (buffer_pos_ == 4) {
    // numpy increments the 256-bit counter BEFORE generating each block, so the
    // first output block is for counter == 1 (not 0).
    if (++counter_[0] == 0)
      if (++counter_[1] == 0)
        if (++counter_[2] == 0) ++counter_[3];
    buffer_ = philox10(counter_, key_);
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
