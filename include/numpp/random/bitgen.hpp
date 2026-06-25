#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace numpp {
namespace random {

// 128-bit integer (clang/gcc). __extension__ silences -Wpedantic on the type.
__extension__ typedef unsigned __int128 u128;

// Bit-exact reimplementation of numpy's SeedSequence (the entropy mixer behind
// every numpy BitGenerator). Reproduces generate_state() word-for-word.
class SeedSequence {
 public:
  explicit SeedSequence(uint64_t seed) { mix_entropy(coerce(seed)); }

  std::vector<uint32_t> generate_state32(size_t n) const {
    uint32_t hash_const = INIT_B;
    std::vector<uint32_t> state(n);
    for (size_t i = 0; i < n; ++i) {
      uint32_t data = pool_[i % pool_.size()];
      data ^= hash_const;
      hash_const *= MULT_B;
      data *= hash_const;
      data ^= data >> XSHIFT;
      state[i] = data;
    }
    return state;
  }
  std::vector<uint64_t> generate_state64(size_t n) const {
    std::vector<uint32_t> s = generate_state32(2 * n);
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint64_t>(s[2 * i]) | (static_cast<uint64_t>(s[2 * i + 1]) << 32);
    return out;
  }

 private:
  static constexpr uint32_t XSHIFT = 16, INIT_A = 0x43b0d7e5u, MULT_A = 0x931e8875u,
                            INIT_B = 0x8b51f9ddu, MULT_B = 0x58f38dedu,
                            MIX_MULT_L = 0xca01f9ddu, MIX_MULT_R = 0x4973f715u;
  std::array<uint32_t, 4> pool_{};

  static std::vector<uint32_t> coerce(uint64_t seed) {
    std::vector<uint32_t> v;
    if (seed == 0) { v.push_back(0); return v; }
    while (seed) { v.push_back(static_cast<uint32_t>(seed & 0xffffffffu)); seed >>= 32; }
    return v;
  }
  static uint32_t mix(uint32_t x, uint32_t y) { uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y; r ^= r >> XSHIFT; return r; }

  void mix_entropy(const std::vector<uint32_t>& entropy) {
    uint32_t hash_const = INIT_A;
    auto hashmix = [&](uint32_t value) { value ^= hash_const; hash_const *= MULT_A; value *= hash_const; value ^= value >> XSHIFT; return value; };
    for (size_t i = 0; i < pool_.size(); ++i) pool_[i] = hashmix(i < entropy.size() ? entropy[i] : 0u);
    for (size_t s = 0; s < pool_.size(); ++s)
      for (size_t d = 0; d < pool_.size(); ++d)
        if (s != d) pool_[d] = mix(pool_[d], hashmix(pool_[s]));
    for (size_t s = pool_.size(); s < entropy.size(); ++s)
      for (size_t d = 0; d < pool_.size(); ++d) pool_[d] = mix(pool_[d], hashmix(entropy[s]));
  }
};

// numpy's PCG64 (XSL-RR 128/64), seeded from SeedSequence. Bit-exact with
// numpy.random.PCG64(seed).random_raw().
class PCG64 {
 public:
  explicit PCG64(uint64_t seed) {
    std::vector<uint64_t> v = SeedSequence(seed).generate_state64(4);
    seed128((static_cast<u128>(v[0]) << 64) | v[1], (static_cast<u128>(v[2]) << 64) | v[3]);
  }

  uint64_t next64() {
    step();
    uint64_t xored = static_cast<uint64_t>(state_ >> 64) ^ static_cast<uint64_t>(state_);
    unsigned rot = static_cast<unsigned>(state_ >> 122);
    return (xored >> rot) | (xored << ((0u - rot) & 63u));
  }
  // Buffered 32-bit output (numpy next_uint32): low half of a draw first, then
  // the high half. The buffer persists in bit-generator state, as in numpy.
  uint32_t next32() {
    if (has32_) { has32_ = false; return buf32_; }
    uint64_t v = next64();
    has32_ = true;
    buf32_ = static_cast<uint32_t>(v >> 32);
    return static_cast<uint32_t>(v & 0xffffffffu);
  }
  // [0,1) double via numpy's 53-bit method.
  double next_double() { return (next64() >> 11) * (1.0 / 9007199254740992.0); }

  uint64_t state_hi() const { return static_cast<uint64_t>(state_ >> 64); }
  uint64_t state_lo() const { return static_cast<uint64_t>(state_); }

 private:
  static constexpr u128 mult() { return (static_cast<u128>(0x2360ed051fc65da4ULL) << 64) | 0x4385df649fccf645ULL; }
  void step() { state_ = state_ * mult() + inc_; }
  void seed128(u128 initstate, u128 initseq) {
    state_ = 0;
    inc_ = (initseq << 1) | 1;
    step();
    state_ += initstate;
    step();
  }
  u128 state_ = 0, inc_ = 0;
  bool has32_ = false;
  uint32_t buf32_ = 0;
};

}  // namespace random
}  // namespace numpp
