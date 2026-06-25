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

inline numpp::DType descr_to_dtype(const std::string& d) {
  // d like "<f8", "|b1", "<i4", "<c16"; byte-order char ignored (native here).
  std::string code = d.substr(d.size() >= 2 ? d.size() - 2 : 0);
  if (d.find("b1") != std::string::npos) return numpp::kBool;
  static const std::pair<const char*, numpp::DType> map[] = {
    {"i1", numpp::kInt8}, {"i2", numpp::kInt16}, {"i4", numpp::kInt32}, {"i8", numpp::kInt64},
    {"u1", numpp::kUInt8}, {"u2", numpp::kUInt16}, {"u4", numpp::kUInt32}, {"u8", numpp::kUInt64},
    {"f2", numpp::kFloat16}, {"f4", numpp::kFloat32}, {"f8", numpp::kFloat64},
    {"c8", numpp::kComplex64}, {"c16", numpp::kComplex128},
  };
  for (auto& [k, v] : map) if (d.find(k) != std::string::npos) return v;
  throw numpp::type_error("oracle: unsupported npy descr " + d);
}

inline numpp::ndarray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < 10 || std::string(buf.data(), 6) != "\x93NUMPY")
    throw numpp::value_error("oracle: bad npy file");
  uint16_t hlen = static_cast<uint8_t>(buf[8]) | (static_cast<uint8_t>(buf[9]) << 8);
  std::string header(buf.data() + 10, hlen);
  auto field = [&](const std::string& key) {
    size_t p = header.find(key);
    return p == std::string::npos ? std::string() : header.substr(p + key.size());
  };
  // descr
  std::string ds = field("'descr':");
  size_t q1 = ds.find('\''), q2 = ds.find('\'', q1 + 1);
  numpp::DType dt = descr_to_dtype(ds.substr(q1 + 1, q2 - q1 - 1));
  // shape
  std::string ss = field("'shape':");
  size_t lp = ss.find('('), rp = ss.find(')');
  numpp::Shape shape;
  std::string inside = ss.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < inside.size();) {
    if (std::isdigit(static_cast<unsigned char>(inside[i]))) {
      int64_t v = 0;
      while (i < inside.size() && std::isdigit(static_cast<unsigned char>(inside[i])))
        v = v * 10 + (inside[i++] - '0');
      shape.push_back(v);
    } else ++i;
  }
  numpp::ndarray out(shape, dt, numpp::Order::C);
  const char* src = buf.data() + 10 + hlen;
  std::memcpy(out.bytes(), src, static_cast<size_t>(out.nbytes()));
  return out;
}

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
