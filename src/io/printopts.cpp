#include "numpp/io/printopts.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <string>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <optional>
#include <vector>
#include <algorithm>


namespace numpp {

namespace {

// Global print state.
PrintOptions g_printopts{};

// Strip trailing zeros after a decimal point. Leaves a trailing '.' (numpy style).
std::string strip_trailing_zeros(std::string s) {
  if (s.find('.') == std::string::npos) return s;
  if (s.find('e') != std::string::npos || s.find('E') != std::string::npos) return s;
  std::size_t last = s.size();
  while (last > 0 && s[last - 1] == '0') --last;
  s.erase(last);
  return s;
}

// Shortest round-tripping %g representation of x.
std::string shortest_g(double x) {
  char buf[64];
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*g", p, x);
    if (std::strtod(buf, nullptr) == x) return std::string(buf);
  }
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return std::string(buf);
}

// Shortest round-tripping %e representation of x.
std::string shortest_e(double x) {
  char buf[64];
  for (int p = 0; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*e", p, x);
    if (std::strtod(buf, nullptr) == x) return std::string(buf);
  }
  std::snprintf(buf, sizeof(buf), "%.17e", x);
  return std::string(buf);
}

bool is_special(const std::string& s) {
  return s.find("inf") != std::string::npos || s.find("nan") != std::string::npos ||
         s.find("Inf") != std::string::npos || s.find("Nan") != std::string::npos ||
         s.find("NaN") != std::string::npos;
}

}  // namespace

void set_printoptions(const PrintOptions& opts) { g_printopts = opts; }

PrintOptions get_printoptions() { return g_printopts; }

std::string format_float_positional(double x, std::optional<int> precision, bool unique,
                                    bool trim_dot) {
  std::string s;
  if (precision.has_value()) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*f", *precision, x);
    s = buf;
  } else {
    s = shortest_g(x);
  }

  if (is_special(s)) return s;

  // Ensure a decimal point is present (numpy always shows one in positional form).
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find('E') == std::string::npos) {
    s += '.';
  }

  // numpy default (unique=true) collapses to the shortest form by stripping
  // trailing zeros; an explicit non-unique precision keeps the fixed digits.
  if (unique) {
    s = strip_trailing_zeros(s);
  }

  if (trim_dot && !s.empty() && s.back() == '.') {
    s.pop_back();
  }

  return s;
}

std::string format_float_scientific(double x, std::optional<int> precision) {
  std::string s;
  if (precision.has_value()) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*e", *precision, x);
    s = buf;
  } else {
    s = shortest_e(x);
    if (!is_special(s)) {
      // Strip trailing zeros in the mantissa while keeping the exponent.
      std::size_t epos = s.find('e');
      if (epos == std::string::npos) epos = s.find('E');
      if (epos != std::string::npos) {
        std::string mant = s.substr(0, epos);
        std::string exp = s.substr(epos);
        if (mant.find('.') != std::string::npos) {
          mant = strip_trailing_zeros(mant);
        }
        s = mant + exp;
      }
    }
  }
  return s;
}

std::string array2string(const ndarray& a, std::optional<int> precision, bool suppress_small) {
  int prec = precision.value_or(g_printopts.precision);
  ndarray c = a.astype(kFloat64).ascontiguousarray();

  auto fmt = [&](double v) -> std::string {
    if (suppress_small && std::abs(v) < 1e-10) v = 0.0;
    return format_float_positional(v, std::nullopt, true, false);
  };
  (void)prec;

  const Shape& shp = c.shape();

  if (shp.empty()) {  // 0-D scalar
    return fmt(c.item<double>({}));
  }

  if (shp.size() == 1) {
    std::string out = "[";
    for (int64_t i = 0; i < shp[0]; ++i) {
      if (i) out += ' ';
      out += fmt(c.item<double>({i}));
    }
    out += ']';
    return out;
  }

  if (shp.size() == 2) {
    std::string out = "[";
    for (int64_t i = 0; i < shp[0]; ++i) {
      if (i) out += "\n ";
      out += '[';
      for (int64_t j = 0; j < shp[1]; ++j) {
        if (j) out += ' ';
        out += fmt(c.item<double>({i, j}));
      }
      out += ']';
    }
    out += ']';
    return out;
  }

  // Higher dimensions: flatten best-effort.
  std::string out = "[";
  int64_t n = c.size();
  ndarray flat = c.ravel();
  for (int64_t i = 0; i < n; ++i) {
    if (i) out += ' ';
    out += fmt(flat.item<double>({i}));
  }
  out += ']';
  return out;
}

}  // namespace numpp
