#pragma once

#include <array>
#include <cstdint>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace random {

// numpy.random.Philox (Philox4x64-10), seeded from SeedSequence; bit-exact raw
// stream with numpy.random.Philox(seed).random_raw().
class NUMPP_API Philox {
 public:
  explicit Philox(uint64_t seed);
  uint64_t next64();
  ndarray random_raw(int64_t n);  // uint64 array

 private:
  std::array<uint64_t, 4> counter_{};
  std::array<uint64_t, 2> key_{};
  std::array<uint64_t, 4> buffer_{};
  int buffer_pos_ = 4;
};

// numpy.random.SFC64, seeded from SeedSequence; bit-exact raw stream.
class NUMPP_API SFC64 {
 public:
  explicit SFC64(uint64_t seed);
  uint64_t next64();
  ndarray random_raw(int64_t n);  // uint64 array

 private:
  uint64_t a_ = 0, b_ = 0, c_ = 0, counter_ = 1;
};

}  // namespace random
}  // namespace numpp
