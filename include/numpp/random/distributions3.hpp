#pragma once

#include <cstdint>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/random/random.hpp"

namespace numpp {
namespace random {

// Remaining numpy distributions (statistical validation; not bit-exact).
NUMPP_API ndarray hypergeometric(Generator& g, int64_t ngood, int64_t nbad, int64_t nsample, const Shape& size);
NUMPP_API ndarray noncentral_f(Generator& g, double dfnum, double dfden, double nonc, const Shape& size);
// Raw random bytes as a uint8 array of the given length (numpy.random.Generator.bytes).
NUMPP_API ndarray bytes(Generator& g, int64_t length);

}  // namespace random
}  // namespace numpp
