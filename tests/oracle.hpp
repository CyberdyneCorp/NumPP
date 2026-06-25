#pragma once
// NumPy validation oracle: evaluate a NumPy expression in Python, load the
// resulting array (.npy) into a numpp::ndarray, and compare. Tests skip (not
// fail) when Python/NumPy is unavailable. Deterministic inputs use fixed seeds.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "numpp/numpp.hpp"

namespace npt {

inline bool numpy_available() {
  static int cached = -1;
  if (cached < 0) cached = std::system("python3 -c \"import numpy\" >/dev/null 2>&1") == 0 ? 1 : 0;
  return cached == 1;
}

// The oracle now uses the production NPY reader (numpp::load) instead of a
// test-only parser, so the loader itself is exercised by every oracle test.
inline numpp::ndarray load_npy(const std::string& path) { return numpp::load(path); }

#ifndef NUMPP_GOLDEN_DIR
#define NUMPP_GOLDEN_DIR "golden"
#endif

// Stable golden-file path for a given oracle expression (keyed by a hash of the
// code, so freezing needs no per-test wiring).
inline std::string golden_path(const std::string& code) {
  size_t h = std::hash<std::string>{}(code);
  return std::string(NUMPP_GOLDEN_DIR) + "/g" + std::to_string(h) + ".npy";
}

// Run NumPy code that assigns variable `a`; return it as an ndarray.
//
// Modes (env):
//   NUMPP_ORACLE_FROZEN=1  -> never call Python; load checked-in golden data
//                             (skip the test if a golden file is missing).
//   NUMPP_ORACLE_RECORD=1  -> run Python live AND freeze the result to golden/.
//   (default)              -> run Python live, skip if NumPy is unavailable.
inline std::optional<numpp::ndarray> oracle(const std::string& code) {
  namespace fs = std::filesystem;
  const bool frozen = std::getenv("NUMPP_ORACLE_FROZEN") != nullptr;
  const bool record = std::getenv("NUMPP_ORACLE_RECORD") != nullptr;
  const fs::path golden = golden_path(code);

  if (frozen) {
    if (!fs::exists(golden)) return std::nullopt;  // skip: no frozen data
    return load_npy(golden.string());
  }
  if (!numpy_available()) return std::nullopt;

  static int counter = 0;
  fs::path dir = fs::temp_directory_path();
  fs::path npy = dir / ("numpp_oracle_" + std::to_string(++counter) + ".npy");
  fs::path py = dir / ("numpp_oracle_" + std::to_string(counter) + ".py");
  {
    std::ofstream s(py);
    s << "import numpy as np\n" << code << "\n"
      << "np.save(r'" << npy.string() << "', np.asarray(a))\n";
  }
  std::string cmd = "python3 " + py.string() + " >/dev/null 2>&1";
  if (std::system(cmd.c_str()) != 0) return std::nullopt;
  auto arr = load_npy(npy.string());
  if (record) {
    std::error_code ec;
    fs::create_directories(fs::path(golden).parent_path(), ec);
    fs::copy_file(npy, golden, fs::copy_options::overwrite_existing, ec);
  }
  std::error_code ec;
  fs::remove(npy, ec);
  fs::remove(py, ec);
  return arr;
}

}  // namespace npt
