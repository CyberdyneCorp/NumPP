#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

using NamedArray = std::pair<std::string, ndarray>;

// NPZ = ZIP archive of <name>.npy members. Interoperates with numpy.savez/np.load.
// savez_compressed currently stores (no DEFLATE) — still readable by numpy.
NUMPP_API void savez(const std::string& path, const std::vector<NamedArray>& arrays);
NUMPP_API void savez_compressed(const std::string& path, const std::vector<NamedArray>& arrays);
NUMPP_API std::map<std::string, ndarray> load_npz(const std::string& path);

}  // namespace numpp
