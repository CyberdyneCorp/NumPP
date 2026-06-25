#include "numpp/io/format.hpp"

#include "numpp/datetime/datetime.hpp"
#include "numpp/io/npy.hpp"
#include "numpp/strings/strings.hpp"
#include "numpp/struct/struct.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

namespace numpp {
namespace {

constexpr int64_t kEdge = 3, kThreshold = 1000;

std::string rjust(const std::string& s, size_t w) { return s.size() >= w ? s : std::string(w - s.size(), ' ') + s; }
std::string ljust(const std::string& s, size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }

std::string fmt_fixed(double v) {  // %.8f with trailing-zero trim, keeping the dot
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.8f", v);
  std::string s = buf;
  size_t dot = s.find('.');
  size_t last = s.size() - 1;
  while (last > dot && s[last] == '0') --last;
  return s.substr(0, last + 1);
}

bool is_default_dtype(DType d) { return d == kInt64 || d == kFloat64 || d == kComplex128 || d == kBool; }

// Produce padded element strings (C order) for the whole array.
// Per-kind element formatters. element_strings() is just a dispatcher.
std::vector<std::string> fmt_string_kind(const ndarray& c, int64_t n) {
  std::vector<std::string> out(n);
  for (int64_t i = 0; i < n; ++i) out[i] = "'" + get_string(c, i) + "'";
  return out;
}
std::vector<std::string> fmt_datetime_kind(const ndarray& c, int64_t n, char k) {
  std::vector<std::string> out(n);
  for (int64_t i = 0; i < n; ++i) {
    std::string v = format_datetime(c.dtype(), dt_get(c, i));
    out[i] = k == 'M' ? "'" + v + "'" : v;
  }
  return out;
}
std::string fmt_field_scalar(const ndarray& v, int64_t i) {
  std::string s;
  visit_dtype(v.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, bool>) s = v.template item<bool>({i}) ? "True" : "False";
    else if constexpr (std::is_integral_v<T>) s = std::to_string(static_cast<long long>(v.template item<T>({i})));
    else if constexpr (std::is_same_v<T, half>) s = fmt_fixed(static_cast<double>(static_cast<float>(v.template item<half>({i}))));
    else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) { auto z = v.template item<T>({i}); s = fmt_fixed(z.real()) + (z.imag() < 0 ? "-" : "+") + fmt_fixed(std::abs(z.imag())) + "j"; }
    else s = fmt_fixed(static_cast<double>(v.template item<T>({i})));
  });
  return s;
}
std::vector<std::string> fmt_struct_kind(const ndarray& c, int64_t n) {
  std::vector<std::string> out(n);
  std::vector<ndarray> fv;
  for (const auto& f : c.dtype().meta()->fields) fv.push_back(field_view(c, f.name));
  for (int64_t i = 0; i < n; ++i) {
    std::string rec = "(";
    for (size_t j = 0; j < fv.size(); ++j) { rec += fmt_field_scalar(fv[j], i); if (j + 1 < fv.size()) rec += ", "; }
    out[i] = rec + ")";
  }
  return out;
}
std::vector<std::string> fmt_int_kind(const ndarray& c, int64_t n) {
  std::vector<std::string> out(n);
  size_t w = 0;
  visit_dtype(c.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    const T* p = n ? c.typed_data<T>() : nullptr;
    for (int64_t i = 0; i < n; ++i) {
      if constexpr (std::is_same_v<T, bool>) out[i] = p[i] ? "True" : "False";
      else if constexpr (std::is_integral_v<T>) out[i] = std::to_string(static_cast<long long>(p[i]));
      w = std::max(w, out[i].size());
    }
  });
  for (auto& s : out) s = rjust(s, w);
  return out;
}
std::vector<std::string> fmt_float_kind(const ndarray& c, int64_t n) {
  std::vector<std::string> out(n), lefts(n), rights(n), specials(n);
  size_t pl = 0, pr = 0;
  auto val = [&](int64_t i) -> double {
    if (c.dtype() == kFloat16) return static_cast<double>(static_cast<float>(c.typed_data<half>()[i]));
    if (c.dtype() == kFloat32) return static_cast<double>(c.typed_data<float>()[i]);
    return c.typed_data<double>()[i];
  };
  for (int64_t i = 0; i < n; ++i) {
    std::string s = fmt_fixed(val(i));
    if (s == "nan" || s == "inf" || s == "-inf") { specials[i] = s; continue; }
    size_t dot = s.find('.');
    lefts[i] = s.substr(0, dot); rights[i] = s.substr(dot + 1);
    pl = std::max(pl, lefts[i].size()); pr = std::max(pr, rights[i].size());
  }
  const size_t w = pl + 1 + pr;
  for (int64_t i = 0; i < n; ++i)
    out[i] = !specials[i].empty() ? rjust(specials[i], w) : rjust(lefts[i], pl) + "." + ljust(rights[i], pr);
  return out;
}
std::vector<std::string> fmt_complex_kind(const ndarray& c, int64_t n) {
  std::vector<std::string> out(n), rL(n), rR(n), iL(n), iR(n), sign(n);
  size_t rpl = 0, rpr = 0, ipl = 0, ipr = 0;
  auto split = [](const std::string& s, size_t& pl, size_t& pr, std::string& L, std::string& R) {
    size_t dot = s.find('.');
    if (dot == std::string::npos) { L = s; R = ""; } else { L = s.substr(0, dot); R = s.substr(dot + 1); }
    pl = std::max(pl, L.size()); pr = std::max(pr, R.size());
  };
  for (int64_t i = 0; i < n; ++i) {
    std::complex<double> z = c.dtype() == kComplex64
        ? std::complex<double>(c.typed_data<std::complex<float>>()[i].real(), c.typed_data<std::complex<float>>()[i].imag())
        : c.typed_data<std::complex<double>>()[i];
    split(fmt_fixed(z.real()), rpl, rpr, rL[i], rR[i]);
    split(fmt_fixed(std::abs(z.imag())), ipl, ipr, iL[i], iR[i]);
    sign[i] = (z.imag() < 0 || (z.imag() == 0 && std::signbit(z.imag()))) ? "-" : "+";
  }
  for (int64_t i = 0; i < n; ++i)
    out[i] = rjust(rL[i], rpl) + "." + ljust(rR[i], rpr) + sign[i] + rjust(iL[i], ipl) + "." + ljust(iR[i], ipr) + "j";
  return out;
}
std::vector<std::string> element_strings(const ndarray& a) {
  ndarray c = a.ascontiguousarray();
  const int64_t n = c.size();
  const char k = c.dtype().kind();
  if (k == 'U' || k == 'S') return fmt_string_kind(c, n);
  if (k == 'M' || k == 'm') return fmt_datetime_kind(c, n, k);
  if (k == 'V') return fmt_struct_kind(c, n);
  if (k == 'i' || k == 'u' || c.dtype() == kBool) return fmt_int_kind(c, n);
  if (k == 'f') return fmt_float_kind(c, n);
  return fmt_complex_kind(c, n);
}

struct Layout {
  const std::vector<std::string>& elems;
  const Shape& shape;
  int64_t ndim;
  std::string sep;     // " " (str) or ", " (repr)
  int base;            // 0 (str) or 6 (repr, for "array(")
  bool summarize;

  std::string rec(int dim, int64_t off, int64_t stride) const {
    const int64_t n = shape[dim];
    const bool sm = summarize && n > 2 * kEdge;
    if (dim == ndim - 1) {
      std::string body;
      auto add = [&](int64_t i, bool last) { body += elems[off + i]; if (!last) body += sep; };
      if (sm) {
        for (int64_t i = 0; i < kEdge; ++i) add(i, false);
        body += "..." + sep;
        for (int64_t i = n - kEdge; i < n; ++i) add(i, i == n - 1);
      } else for (int64_t i = 0; i < n; ++i) add(i, i == n - 1);
      return "[" + body + "]";
    }
    int64_t sub_stride = stride / n;
    int hang = base + dim + 1;
    std::string seph = (sep == ", " ? "," : "");
    seph += std::string(static_cast<size_t>(ndim - 1 - dim), '\n') + std::string(static_cast<size_t>(hang), ' ');
    std::string body;
    auto add = [&](int64_t i, bool last) { body += rec(dim + 1, off + i * sub_stride, sub_stride); if (!last) body += seph; };
    if (sm) {
      for (int64_t i = 0; i < kEdge; ++i) add(i, false);
      body += "..." + seph;
      for (int64_t i = n - kEdge; i < n; ++i) add(i, i == n - 1);
    } else for (int64_t i = 0; i < n; ++i) add(i, i == n - 1);
    return "[" + body + "]";
  }
};

std::string format_body(const ndarray& a, const std::string& sep, int base) {
  if (a.ndim() == 0) { auto e = element_strings(a); return e.empty() ? "" : e[0]; }
  std::vector<std::string> elems = element_strings(a);
  Layout L{elems, a.shape(), a.ndim(), sep, base, a.size() > kThreshold};
  return L.rec(0, 0, a.size());
}

}  // namespace

std::string array_str(const ndarray& a) {
  if (a.ndim() == 0) { auto e = element_strings(a); return e.empty() ? "" : e[0]; }
  return format_body(a, " ", 0);
}

std::string array_repr(const ndarray& a) {
  std::string body;
  if (a.size() == 0) body = "[]";
  else body = a.ndim() == 0 ? format_body(a, ", ", 6) : format_body(a, ", ", 6);
  if (a.dtype().is_extended()) {
    const char k = a.dtype().kind();
    std::string suffix;
    if (k == 'M') suffix = std::string("'datetime64[") + a.dtype().meta()->unit + "]'";
    else if (k == 'm') suffix = std::string("'timedelta64[") + a.dtype().meta()->unit + "]'";
    else if (k == 'V') {
      suffix = "[";
      const auto& fields = a.dtype().meta()->fields;
      for (size_t j = 0; j < fields.size(); ++j) { suffix += "('" + fields[j].name + "', '" + dtype_to_descr(fields[j].dtype) + "')"; if (j + 1 < fields.size()) suffix += ", "; }
      suffix += "]";
    } else suffix = "'" + dtype_to_descr(a.dtype()) + "'";
    return "array(" + body + ", dtype=" + suffix + ")";
  }
  if (!is_default_dtype(a.dtype()))
    return "array(" + body + ", dtype=" + a.dtype().name() + ")";
  return "array(" + body + ")";
}

}  // namespace numpp
