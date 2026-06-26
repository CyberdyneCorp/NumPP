#include "numpp/strings/char_extra2.hpp"

#include "numpp/strings/strings.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <string>
#include <vector>


namespace numpp {

namespace npchar {
namespace {

std::vector<std::string> read_all_cc(const ndarray& a) {
  const int64_t n = a.size();
  std::vector<std::string> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) out.push_back(get_string(a, i));
  return out;
}

}  // namespace

ndarray join(const std::string& sep, const ndarray& a) {
  std::vector<std::string> in = read_all_cc(a);
  std::vector<std::string> out;
  out.reserve(in.size());
  for (const auto& s : in) {
    std::string r;
    for (size_t k = 0; k < s.size(); ++k) {
      if (k != 0) r += sep;
      r += s[k];
    }
    out.push_back(r);
  }
  return string_array(out);
}

ndarray encode(const ndarray& a) {
  // 'U' -> 'S'; ASCII element strings are unchanged.
  return bytes_array(read_all_cc(a));
}

ndarray decode(const ndarray& a) {
  // 'S' -> 'U'; ASCII element strings are unchanged.
  return string_array(read_all_cc(a));
}

Partition3 partition(const ndarray& a, const std::string& sep) {
  std::vector<std::string> in = read_all_cc(a);
  std::vector<std::string> before, mid, after;
  before.reserve(in.size());
  mid.reserve(in.size());
  after.reserve(in.size());
  for (const auto& s : in) {
    const size_t pos = sep.empty() ? std::string::npos : s.find(sep);
    if (pos == std::string::npos) {
      before.push_back(s);
      mid.push_back(std::string());
      after.push_back(std::string());
    } else {
      before.push_back(s.substr(0, pos));
      mid.push_back(sep);
      after.push_back(s.substr(pos + sep.size()));
    }
  }
  return Partition3{string_array(before), string_array(mid), string_array(after)};
}

}  // namespace npchar

}  // namespace numpp
