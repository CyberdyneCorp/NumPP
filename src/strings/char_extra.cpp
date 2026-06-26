#include "numpp/strings/char_extra.hpp"

#include "numpp/strings/strings.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <functional>


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

ndarray map_strings(const ndarray& a,
                    const std::function<std::string(const std::string&)>& f) {
  std::vector<std::string> in = read_all(a);
  std::vector<std::string> out;
  out.reserve(in.size());
  for (const auto& s : in) out.push_back(f(s));
  return string_array(out);
}

ndarray map_bool(const ndarray& a,
                 const std::function<bool(const std::string&)>& f) {
  std::vector<std::string> in = read_all(a);
  ndarray out({static_cast<int64_t>(in.size())}, kBool, Order::C);
  for (size_t i = 0; i < in.size(); ++i) out.typed_data<bool>()[i] = f(in[i]);
  return out;
}

std::string pad_center(const std::string& s, int64_t width, char fill) {
  const int64_t len = static_cast<int64_t>(s.size());
  if (len >= width) return s;
  const int64_t marg = width - len;
  const int64_t left = marg / 2;      // extra padding goes on the right
  const int64_t right = marg - left;
  return std::string(static_cast<size_t>(left), fill) + s +
         std::string(static_cast<size_t>(right), fill);
}

std::string pad_left(const std::string& s, int64_t width, char fill) {
  const int64_t len = static_cast<int64_t>(s.size());
  if (len >= width) return s;
  return std::string(static_cast<size_t>(width - len), fill) + s;
}

std::string pad_right(const std::string& s, int64_t width, char fill) {
  const int64_t len = static_cast<int64_t>(s.size());
  if (len >= width) return s;
  return s + std::string(static_cast<size_t>(width - len), fill);
}

std::string zfill_s(const std::string& s, int64_t width) {
  const int64_t len = static_cast<int64_t>(s.size());
  if (len >= width) return s;
  std::string sign;
  std::string rest = s;
  if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
    sign = s.substr(0, 1);
    rest = s.substr(1);
  }
  return sign + std::string(static_cast<size_t>(width - len), '0') + rest;
}

std::string swapcase_s(const std::string& s) {
  std::string r = s;
  for (char& c : r) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::islower(uc))
      c = static_cast<char>(std::toupper(uc));
    else if (std::isupper(uc))
      c = static_cast<char>(std::tolower(uc));
  }
  return r;
}

std::string expandtabs_s(const std::string& s, int64_t tabsize) {
  std::string r;
  int64_t col = 0;
  for (char c : s) {
    if (c == '\t') {
      if (tabsize > 0) {
        const int64_t fill = tabsize - (col % tabsize);
        r.append(static_cast<size_t>(fill), ' ');
        col += fill;
      }
    } else if (c == '\n' || c == '\r') {
      r.push_back(c);
      col = 0;
    } else {
      r.push_back(c);
      ++col;
    }
  }
  return r;
}

bool all_of_pred(const std::string& s, int (*pred)(int)) {
  if (s.empty()) return false;
  for (char c : s)
    if (!pred(static_cast<unsigned char>(c))) return false;
  return true;
}

bool is_upper_s(const std::string& s) {
  bool has_cased = false;
  for (char c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::islower(uc)) return false;
    if (std::isupper(uc)) has_cased = true;
  }
  return has_cased;
}

bool is_lower_s(const std::string& s) {
  bool has_cased = false;
  for (char c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isupper(uc)) return false;
    if (std::islower(uc)) has_cased = true;
  }
  return has_cased;
}

bool is_title_s(const std::string& s) {
  bool cased = false;
  bool prev_cased = false;
  for (char c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isupper(uc)) {
      if (prev_cased) return false;
      prev_cased = true;
      cased = true;
    } else if (std::islower(uc)) {
      if (!prev_cased) return false;
      prev_cased = true;
      cased = true;
    } else {
      prev_cased = false;
    }
  }
  return cased;
}

}  // namespace

ndarray center(const ndarray& a, int64_t width, char fillchar) {
  return map_strings(a, [=](const std::string& s) { return pad_center(s, width, fillchar); });
}
ndarray ljust(const ndarray& a, int64_t width, char fillchar) {
  return map_strings(a, [=](const std::string& s) { return pad_right(s, width, fillchar); });
}
ndarray rjust(const ndarray& a, int64_t width, char fillchar) {
  return map_strings(a, [=](const std::string& s) { return pad_left(s, width, fillchar); });
}
ndarray zfill(const ndarray& a, int64_t width) {
  return map_strings(a, [=](const std::string& s) { return zfill_s(s, width); });
}
ndarray swapcase(const ndarray& a) { return map_strings(a, swapcase_s); }
ndarray expandtabs(const ndarray& a, int64_t tabsize) {
  return map_strings(a, [=](const std::string& s) { return expandtabs_s(s, tabsize); });
}

ndarray isalpha(const ndarray& a) {
  return map_bool(a, [](const std::string& s) { return all_of_pred(s, std::isalpha); });
}
ndarray isdigit(const ndarray& a) {
  return map_bool(a, [](const std::string& s) { return all_of_pred(s, std::isdigit); });
}
ndarray isspace(const ndarray& a) {
  return map_bool(a, [](const std::string& s) { return all_of_pred(s, std::isspace); });
}
ndarray isupper(const ndarray& a) { return map_bool(a, is_upper_s); }
ndarray islower(const ndarray& a) { return map_bool(a, is_lower_s); }
ndarray isalnum(const ndarray& a) {
  return map_bool(a, [](const std::string& s) { return all_of_pred(s, std::isalnum); });
}
ndarray istitle(const ndarray& a) { return map_bool(a, is_title_s); }

}  // namespace npchar

}  // namespace numpp
