#include "numpp/strings/char.hpp"

#include "numpp/core/creation.hpp"   // zeros
#include "numpp/strings/strings.hpp" // string_array / get_string

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

namespace numpp {
namespace npchar {
namespace {

std::vector<std::string> read_all(const ndarray& a) {
  const int64_t n = a.size();
  std::vector<std::string> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) out.push_back(get_string(a, i));
  return out;
}

ndarray map_strings(const ndarray& a, const std::function<std::string(const std::string&)>& f) {
  std::vector<std::string> in = read_all(a);
  std::vector<std::string> out;
  out.reserve(in.size());
  for (const auto& s : in) out.push_back(f(s));
  return string_array(out);
}

ndarray map_int(const ndarray& a, const std::function<int64_t(const std::string&)>& f) {
  std::vector<std::string> in = read_all(a);
  ndarray out({static_cast<int64_t>(in.size())}, kInt64, Order::C);
  for (size_t i = 0; i < in.size(); ++i) out.typed_data<int64_t>()[i] = f(in[i]);
  return out;
}
ndarray map_bool(const ndarray& a, const std::function<bool(const std::string&)>& f) {
  std::vector<std::string> in = read_all(a);
  ndarray out({static_cast<int64_t>(in.size())}, kBool, Order::C);
  for (size_t i = 0; i < in.size(); ++i) out.typed_data<bool>()[i] = f(in[i]);
  return out;
}

std::string lstrip_s(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n\f\v");
  return a == std::string::npos ? std::string() : s.substr(a);
}
std::string rstrip_s(const std::string& s) {
  size_t b = s.find_last_not_of(" \t\r\n\f\v");
  return b == std::string::npos ? std::string() : s.substr(0, b + 1);
}
int64_t count_sub(const std::string& s, const std::string& sub) {
  if (sub.empty()) return static_cast<int64_t>(s.size()) + 1;
  int64_t c = 0;
  size_t pos = 0;
  while ((pos = s.find(sub, pos)) != std::string::npos) { ++c; pos += sub.size(); }
  return c;
}

}  // namespace

ndarray add(const ndarray& a, const ndarray& b) {
  std::vector<std::string> av = read_all(a), bv = read_all(b);
  if (av.size() != bv.size()) throw value_error("char.add: length mismatch");
  std::vector<std::string> out(av.size());
  for (size_t i = 0; i < av.size(); ++i) out[i] = av[i] + bv[i];
  return string_array(out);
}
ndarray multiply(const ndarray& a, int64_t n) {
  return map_strings(a, [n](const std::string& s) {
    std::string r;
    for (int64_t k = 0; k < n; ++k) r += s;
    return r;
  });
}
ndarray upper(const ndarray& a) {
  return map_strings(a, [](const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::toupper(c); });
    return r;
  });
}
ndarray lower(const ndarray& a) {
  return map_strings(a, [](const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
  });
}
ndarray capitalize(const ndarray& a) {
  return map_strings(a, [](const std::string& s) {
    std::string r = s;
    for (size_t i = 0; i < r.size(); ++i)
      r[i] = static_cast<char>(i == 0 ? std::toupper(static_cast<unsigned char>(r[i]))
                                      : std::tolower(static_cast<unsigned char>(r[i])));
    return r;
  });
}
ndarray title(const ndarray& a) {
  return map_strings(a, [](const std::string& s) {
    std::string r = s;
    bool start = true;
    for (char& c : r) {
      const unsigned char uc = static_cast<unsigned char>(c);
      if (std::isalpha(uc)) { c = static_cast<char>(start ? std::toupper(uc) : std::tolower(uc)); start = false; }
      else start = true;
    }
    return r;
  });
}
ndarray strip(const ndarray& a) { return map_strings(a, [](const std::string& s) { return rstrip_s(lstrip_s(s)); }); }
ndarray lstrip(const ndarray& a) { return map_strings(a, lstrip_s); }
ndarray rstrip(const ndarray& a) { return map_strings(a, rstrip_s); }
ndarray replace(const ndarray& a, const std::string& old_s, const std::string& new_s) {
  return map_strings(a, [&](const std::string& s) {
    if (old_s.empty()) return s;
    std::string r = s;
    size_t pos = 0;
    while ((pos = r.find(old_s, pos)) != std::string::npos) { r.replace(pos, old_s.size(), new_s); pos += new_s.size(); }
    return r;
  });
}

ndarray str_len(const ndarray& a) { return map_int(a, [](const std::string& s) { return static_cast<int64_t>(s.size()); }); }
ndarray find(const ndarray& a, const std::string& sub) {
  return map_int(a, [&](const std::string& s) {
    const size_t p = s.find(sub);
    return p == std::string::npos ? int64_t{-1} : static_cast<int64_t>(p);
  });
}
ndarray count(const ndarray& a, const std::string& sub) { return map_int(a, [&](const std::string& s) { return count_sub(s, sub); }); }
ndarray startswith(const ndarray& a, const std::string& prefix) {
  return map_bool(a, [&](const std::string& s) { return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0; });
}
ndarray endswith(const ndarray& a, const std::string& suffix) {
  return map_bool(a, [&](const std::string& s) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  });
}

}  // namespace npchar
}  // namespace numpp
