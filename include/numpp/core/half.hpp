#pragma once

#include <cstdint>
#include <cstring>

namespace numpp {

// Minimal IEEE-754 binary16 stored in a uint16_t. Conversions to/from float are
// exact for representable values (round-to-nearest-even on narrowing). This is a
// clean-room implementation; NumPy's npymath/halffloat can be substituted later
// for the SIMD-accelerated paths.
struct half {
  uint16_t bits = 0;

  half() = default;
  explicit half(float f) : bits(from_float(f)) {}
  explicit operator float() const { return to_float(bits); }

  static uint16_t from_float(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {            // inf / nan
      return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (exp <= 0) {                               // subnormal / underflow
      if (exp < -10) return static_cast<uint16_t>(sign);
      uint32_t m = (mant | 0x800000u) >> (1 - exp);
      if (m & 0x1000u) m += 0x2000u;              // round to nearest even
      return static_cast<uint16_t>(sign | (m >> 13));
    }
    uint32_t out = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) out += 1;                 // round to nearest even
    return static_cast<uint16_t>(out);
  }

  static float to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
      if (mant == 0) {
        out = sign;                               // +/- zero
      } else {                                    // subnormal
        int32_t e = -1;
        uint32_t m = mant;
        do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
        m &= 0x3FFu;
        out = sign | (static_cast<uint32_t>(127 - 15 + e) << 23) | (m << 13);
      }
    } else if (exp == 0x1F) {
      out = sign | 0x7F800000u | (mant << 13);    // inf / nan
    } else {
      out = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
  }
};

}  // namespace numpp
