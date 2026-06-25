#pragma once
// Shared parity harness for the examples: compute something with NumPP, compute
// the same thing with real NumPy at runtime, and assert they agree. Each example
// prints PASS/FAIL lines and returns ex::summary() as its exit code.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "numpp/numpp.hpp"

namespace ex {

inline int& passed() { static int p = 0; return p; }
inline int& failed() { static int f = 0; return f; }

// Run NumPy code that assigns a result into variable `a`; return it as an ndarray.
inline numpp::ndarray numpy(const std::string& code) {
  namespace fs = std::filesystem;
  static int counter = 0;
  fs::path dir = fs::temp_directory_path();
  fs::path npy = dir / ("numpp_ex_" + std::to_string(++counter) + ".npy");
  fs::path py = dir / ("numpp_ex_" + std::to_string(counter) + ".py");
  {
    std::ofstream s(py);
    s << "import numpy as np\n" << code << "\n"
      << "np.save(r'" << npy.string() << "', np.asarray(a))\n";
  }
  if (std::system(("python3 " + py.string() + " >/dev/null 2>&1").c_str()) != 0)
    throw std::runtime_error("numpy oracle failed for: " + code);
  numpp::ndarray r = numpp::load(npy.string());
  std::error_code ec; fs::remove(npy, ec); fs::remove(py, ec);
  return r;
}

// Compare a NumPP result against a NumPy expression. Prints a PASS/FAIL line.
inline bool check(const std::string& label, const numpp::ndarray& got,
                  const std::string& numpy_code, double rtol = 1e-6, double atol = 1e-8) {
  bool ok = false;
  try {
    numpp::ndarray want = numpy(numpy_code);
    ok = got.shape() == want.shape() && numpp::allclose(got, want, rtol, atol, /*equal_nan=*/true);
  } catch (const std::exception& e) {
    std::printf("  [ERR ] %-44s  %s\n", label.c_str(), e.what());
    ++failed();
    return false;
  }
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", label.c_str());
  (ok ? passed() : failed())++;
  return ok;
}

// Compare against a scalar value (for single-number results).
inline bool check_scalar(const std::string& label, double got, double want, double tol = 1e-6) {
  bool ok = std::abs(got - want) <= tol + tol * std::abs(want);
  std::printf("  [%s] %s  (got %.8g, want %.8g)\n", ok ? "PASS" : "FAIL", label.c_str(), got, want);
  (ok ? passed() : failed())++;
  return ok;
}

inline int summary() {
  std::printf("  ----\n  %d passed, %d failed\n", passed(), failed());
  return failed() == 0 ? 0 : 1;
}

}  // namespace ex
