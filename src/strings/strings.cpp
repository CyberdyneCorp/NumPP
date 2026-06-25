#include "numpp/strings/strings.hpp"

#include <algorithm>
#include <cstring>

namespace numpp {
namespace {

void utf8_encode(uint32_t cp, std::string& out) {
  if (cp < 0x80) out += static_cast<char>(cp);
  else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
  else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
}
std::vector<uint32_t> utf8_decode(const std::string& s) {
  std::vector<uint32_t> cps;
  for (size_t i = 0; i < s.size();) {
    uint8_t c = s[i];
    uint32_t cp; int extra;
    if (c < 0x80) { cp = c; extra = 0; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; extra = 1; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; extra = 2; }
    else { cp = c & 0x07; extra = 3; }
    ++i;
    for (int k = 0; k < extra && i < s.size(); ++k, ++i) cp = (cp << 6) | (s[i] & 0x3F);
    cps.push_back(cp);
  }
  return cps;
}

int64_t max_len_u32(const std::vector<std::string>& strs) {
  int64_t m = 0;
  for (const auto& s : strs) m = std::max(m, static_cast<int64_t>(utf8_decode(s).size()));
  return std::max<int64_t>(m, 1);
}

}  // namespace

ndarray string_array(const std::vector<std::string>& strs, std::optional<int64_t> num_chars) {
  const int64_t n = num_chars ? *num_chars : max_len_u32(strs);
  ndarray a(Shape{static_cast<int64_t>(strs.size())}, make_string(n), Order::C);
  std::memset(a.bytes(), 0, static_cast<size_t>(a.nbytes()));
  for (size_t i = 0; i < strs.size(); ++i) set_string(a, static_cast<int64_t>(i), strs[i]);
  return a;
}
ndarray bytes_array(const std::vector<std::string>& strs, std::optional<int64_t> num_bytes) {
  int64_t n = num_bytes ? *num_bytes : 1;
  if (!num_bytes) for (const auto& s : strs) n = std::max(n, static_cast<int64_t>(s.size()));
  ndarray a(Shape{static_cast<int64_t>(strs.size())}, make_bytes(n), Order::C);
  std::memset(a.bytes(), 0, static_cast<size_t>(a.nbytes()));
  for (size_t i = 0; i < strs.size(); ++i) set_string(a, static_cast<int64_t>(i), strs[i]);
  return a;
}

std::string get_string(const ndarray& a, int64_t i) {
  const int64_t es = a.itemsize();
  const char* p = a.bytes() + i * es;
  std::string out;
  if (a.dtype().kind() == 'U') {
    const int64_t nch = es / 4;
    for (int64_t k = 0; k < nch; ++k) {
      uint32_t cp;
      std::memcpy(&cp, p + 4 * k, 4);
      if (cp == 0) break;
      utf8_encode(cp, out);
    }
  } else {  // 'S'
    for (int64_t k = 0; k < es; ++k) { if (p[k] == 0) break; out += p[k]; }
  }
  return out;
}
void set_string(ndarray& a, int64_t i, const std::string& s) {
  const int64_t es = a.itemsize();
  char* p = a.bytes() + i * es;
  std::memset(p, 0, static_cast<size_t>(es));
  if (a.dtype().kind() == 'U') {
    std::vector<uint32_t> cps = utf8_decode(s);
    const int64_t nch = es / 4;
    for (int64_t k = 0; k < nch && k < static_cast<int64_t>(cps.size()); ++k) std::memcpy(p + 4 * k, &cps[k], 4);
  } else {
    const int64_t m = std::min<int64_t>(es, static_cast<int64_t>(s.size()));
    std::memcpy(p, s.data(), static_cast<size_t>(m));
  }
}

ndarray str_equal(const ndarray& a, const ndarray& b) {
  const int64_t n = a.size();
  ndarray out(a.shape(), kBool, Order::C);
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = get_string(a, i) == get_string(b, i);
  return out;
}
ndarray str_not_equal(const ndarray& a, const ndarray& b) {
  ndarray e = str_equal(a, b);
  const int64_t n = e.size();
  bool* o = n ? e.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = !o[i];
  return e;
}

}  // namespace numpp
