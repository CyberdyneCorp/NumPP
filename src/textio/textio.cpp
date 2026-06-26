#include "numpp/textio/textio.hpp"

#include "numpp/core/creation.hpp"   // zeros

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace numpp {
namespace {

std::vector<std::string> split_fields(const std::string& line, char delimiter) {
  std::vector<std::string> out;
  if (delimiter == '\0') {  // whitespace-separated
    std::istringstream is(line);
    std::string tok;
    while (is >> tok) out.push_back(tok);
  } else {
    std::string cur;
    for (char c : line) {
      if (c == delimiter) { out.push_back(cur); cur.clear(); }
      else cur += c;
    }
    out.push_back(cur);
  }
  return out;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// Parse the file into a row-major double table; returns rows and sets ncols.
std::vector<std::vector<double>> parse_table(const std::string& path, char delimiter, bool nan_on_fail) {
  std::ifstream in(path);
  if (!in) throw value_error("loadtxt: cannot open file");
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.empty() || t[0] == '#') continue;
    std::vector<double> row;
    for (const std::string& f : split_fields(t, delimiter)) {
      const std::string ft = trim(f);
      try {
        size_t pos = 0;
        const double v = std::stod(ft, &pos);
        row.push_back(pos == ft.size() ? v : (nan_on_fail ? std::numeric_limits<double>::quiet_NaN() : throw 0));
      } catch (...) {
        if (nan_on_fail) row.push_back(std::numeric_limits<double>::quiet_NaN());
        else throw value_error("loadtxt: could not parse a value");
      }
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

ndarray table_to_array(const std::vector<std::vector<double>>& rows) {
  const int64_t nrows = static_cast<int64_t>(rows.size());
  const int64_t ncols = nrows ? static_cast<int64_t>(rows[0].size()) : 0;
  if (nrows == 1) {  // single row -> 1-D
    ndarray out({ncols}, kFloat64, Order::C);
    for (int64_t j = 0; j < ncols; ++j) out.typed_data<double>()[j] = rows[0][j];
    return out;
  }
  if (ncols == 1) {  // single column -> 1-D
    ndarray out({nrows}, kFloat64, Order::C);
    for (int64_t i = 0; i < nrows; ++i) out.typed_data<double>()[i] = rows[i][0];
    return out;
  }
  ndarray out({nrows, ncols}, kFloat64, Order::C);
  for (int64_t i = 0; i < nrows; ++i)
    for (int64_t j = 0; j < ncols; ++j) out.typed_data<double>()[i * ncols + j] = rows[i][j];
  return out;
}

}  // namespace

ndarray loadtxt(const std::string& path, char delimiter) {
  return table_to_array(parse_table(path, delimiter, false));
}
ndarray genfromtxt(const std::string& path, char delimiter) {
  return table_to_array(parse_table(path, delimiter, true));
}

void savetxt(const std::string& path, const ndarray& a, const std::string& fmt, const std::string& delimiter) {
  ndarray m = a.astype(kFloat64).ascontiguousarray();
  const int64_t nrows = m.ndim() >= 1 ? m.shape()[0] : 1;
  const int64_t ncols = m.ndim() >= 2 ? m.shape()[1] : (m.ndim() == 1 ? 1 : 1);
  const double* p = m.size() ? m.typed_data<double>() : nullptr;
  std::ofstream out(path);
  if (!out) throw value_error("savetxt: cannot open file");
  char buf[64];
  for (int64_t i = 0; i < nrows; ++i) {
    for (int64_t j = 0; j < ncols; ++j) {
      std::snprintf(buf, sizeof(buf), fmt.c_str(), p[i * ncols + j]);
      out << buf;
      if (j + 1 < ncols) out << delimiter;
    }
    out << "\n";
  }
}

ndarray fromstring(const std::string& s, const std::string& sep) {
  std::vector<double> vals;
  std::string cur;
  auto flush = [&]() {
    const std::string t = trim(cur);
    if (!t.empty()) { try { vals.push_back(std::stod(t)); } catch (...) {} }
    cur.clear();
  };
  for (char c : s) {
    if (!sep.empty() && c == sep[0]) flush();
    else cur += c;
  }
  flush();
  ndarray out({static_cast<int64_t>(vals.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < vals.size(); ++i) out.typed_data<double>()[i] = vals[i];
  return out;
}

void tofile(const ndarray& a, const std::string& path) {
  ndarray m = a.ascontiguousarray();
  std::ofstream out(path, std::ios::binary);
  if (!out) throw value_error("tofile: cannot open file");
  out.write(m.bytes(), m.size() * m.dtype().itemsize());
}

ndarray fromfile(const std::string& path, DType dtype, int64_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw value_error("fromfile: cannot open file");
  const int64_t isz = dtype.itemsize();
  const int64_t bytes_total = static_cast<int64_t>(in.tellg());
  int64_t n = count >= 0 ? count : bytes_total / isz;
  in.seekg(0);
  ndarray out({n}, dtype, Order::C);
  in.read(out.bytes(), n * isz);
  return out;
}

std::string binary_repr(int64_t value, int64_t width) {
  if (value == 0) {
    std::string z(std::max<int64_t>(1, width), '0');
    return z;
  }
  if (value > 0) {
    std::string s;
    for (int64_t v = value; v > 0; v >>= 1) s += static_cast<char>('0' + (v & 1));
    std::reverse(s.begin(), s.end());
    while (static_cast<int64_t>(s.size()) < width) s.insert(s.begin(), '0');
    return s;
  }
  // negative: two's complement in `width` bits (numpy requires width)
  if (width <= 0) return "-" + binary_repr(-value, 0);
  const uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1);
  const uint64_t tc = static_cast<uint64_t>(value) & mask;
  std::string s;
  for (int64_t i = width - 1; i >= 0; --i) s += static_cast<char>('0' + ((tc >> i) & 1));
  return s;
}

std::string base_repr(int64_t value, int64_t base, int64_t padding) {
  if (base < 2 || base > 36) throw value_error("base_repr: base must be in [2, 36]");
  const bool neg = value < 0;
  uint64_t v = neg ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
  const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string s;
  if (v == 0) s = "0";
  while (v > 0) { s += digits[v % static_cast<uint64_t>(base)]; v /= static_cast<uint64_t>(base); }
  for (int64_t i = 0; i < padding; ++i) s += '0';
  std::reverse(s.begin(), s.end());
  return neg ? "-" + s : s;
}

}  // namespace numpp
