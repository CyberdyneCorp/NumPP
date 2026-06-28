#include "numpp/io/format.hpp"

#include "numpp/datetime/datetime.hpp"
#include "numpp/io/npy.hpp"
#include "numpp/strings/strings.hpp"
#include "numpp/struct/struct.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
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
// numpy switches a float column to scientific notation (FloatingFormat) based on
// the finite, nonzero magnitudes: when the largest is >= 1e8, the smallest is
// < 1e-4, or their ratio exceeds 1000.
bool float_use_sci(const std::vector<double>& vals) {
  double mx = 0, mn = 0;
  bool any = false;
  for (double v : vals) {
    if (!std::isfinite(v) || v == 0.0) continue;
    const double a = std::fabs(v);
    if (!any) { mx = mn = a; any = true; }
    else { mx = std::max(mx, a); mn = std::min(mn, a); }
  }
  if (!any) return false;
  return mx >= 1e8 || mn < 1e-4 || mx / mn > 1000.0;
}

// Shortest round-tripping "%e" with at most `maxp` fractional digits (numpy's
// default maxprec floatmode: shortest unique, capped at precision).
std::string shortest_e_capped(double x, int maxp) {
  char buf[64];
  for (int p = 0; p <= maxp; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*e", p, x);
    if (std::strtod(buf, nullptr) == x) return buf;
  }
  std::snprintf(buf, sizeof(buf), "%.*e", maxp, x);
  return buf;
}

// Format a column of doubles in aligned scientific notation, matching numpy:
// one leading digit, a common fractional width, a signed exponent of common
// width (>= 2 digits), and right-justification across the column.
std::vector<std::string> fmt_float_sci(const std::vector<double>& vals) {
  constexpr int kPrec = 8;
  const size_t n = vals.size();
  std::vector<std::string> sign(n), lead(n), frac(n), esign(n), edig(n), special(n);
  size_t fracw = 0, edigw = 2;
  for (size_t i = 0; i < n; ++i) {
    const double v = vals[i];
    if (std::isnan(v)) { special[i] = "nan"; continue; }
    if (std::isinf(v)) { special[i] = v < 0 ? "-inf" : "inf"; continue; }
    sign[i] = std::signbit(v) ? "-" : "";
    const std::string e = shortest_e_capped(std::fabs(v), kPrec);
    const size_t ep = e.find('e');
    const std::string m = e.substr(0, ep), ex = e.substr(ep + 1);
    const size_t dot = m.find('.');
    lead[i] = dot == std::string::npos ? m : m.substr(0, dot);
    frac[i] = dot == std::string::npos ? "" : m.substr(dot + 1);
    esign[i] = ex[0] == '-' ? "-" : "+";
    const size_t exi = (ex[0] == '+' || ex[0] == '-') ? 1 : 0;
    const std::string ed = ex.substr(exi);
    const size_t nz = ed.find_first_not_of('0');
    edig[i] = nz == std::string::npos ? "0" : ed.substr(nz);
    fracw = std::max(fracw, frac[i].size());
    edigw = std::max(edigw, edig[i].size());
  }
  std::vector<std::string> out(n);
  size_t w = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!special[i].empty()) out[i] = special[i];
    else {
      std::string fr = frac[i]; fr.resize(fracw, '0');
      std::string ed = std::string(edigw - edig[i].size(), '0') + edig[i];
      out[i] = sign[i] + lead[i] + "." + fr + "e" + esign[i] + ed;
    }
    w = std::max(w, out[i].size());
  }
  for (auto& s : out) s = rjust(s, w);
  return out;
}

// Format a column of doubles the way numpy does: scientific (aligned) when the
// magnitudes warrant it, otherwise fixed with an aligned decimal point. Shared by
// the real float kind and by each part (real / imag magnitude) of the complex kind.
std::vector<std::string> fmt_double_column(const std::vector<double>& vals) {
  if (float_use_sci(vals)) return fmt_float_sci(vals);
  const size_t n = vals.size();
  std::vector<std::string> out(n), lefts(n), rights(n), specials(n);
  size_t pl = 0, pr = 0;
  for (size_t i = 0; i < n; ++i) {
    std::string s = fmt_fixed(vals[i]);
    if (s == "nan" || s == "inf" || s == "-inf") { specials[i] = s; continue; }
    size_t dot = s.find('.');
    lefts[i] = s.substr(0, dot); rights[i] = s.substr(dot + 1);
    pl = std::max(pl, lefts[i].size()); pr = std::max(pr, rights[i].size());
  }
  const size_t w = pl + 1 + pr;
  for (size_t i = 0; i < n; ++i)
    out[i] = !specials[i].empty() ? rjust(specials[i], w) : rjust(lefts[i], pl) + "." + ljust(rights[i], pr);
  return out;
}

std::vector<std::string> fmt_float_kind(const ndarray& c, int64_t n) {
  auto val = [&](int64_t i) -> double {
    if (c.dtype() == kFloat16) return static_cast<double>(static_cast<float>(c.typed_data<half>()[i]));
    if (c.dtype() == kFloat32) return static_cast<double>(c.typed_data<float>()[i]);
    return c.typed_data<double>()[i];
  };
  std::vector<double> vals(n);
  for (int64_t i = 0; i < n; ++i) vals[i] = val(i);
  return fmt_double_column(vals);
}
std::vector<std::string> fmt_complex_kind(const ndarray& c, int64_t n) {
  // numpy formats the real parts and the imaginary magnitudes as two independent
  // columns (each may pick fixed or scientific on its own), then joins them with
  // the imaginary sign and a trailing 'j'.
  std::vector<double> reals(n), imags(n);
  std::vector<std::string> sign(n), out(n);
  for (int64_t i = 0; i < n; ++i) {
    std::complex<double> z = c.dtype() == kComplex64
        ? std::complex<double>(c.typed_data<std::complex<float>>()[i].real(), c.typed_data<std::complex<float>>()[i].imag())
        : c.typed_data<std::complex<double>>()[i];
    reals[i] = z.real();
    imags[i] = std::abs(z.imag());
    sign[i] = (z.imag() < 0 || (z.imag() == 0 && std::signbit(z.imag()))) ? "-" : "+";
  }
  std::vector<std::string> rcol = fmt_double_column(reals);
  std::vector<std::string> icol = fmt_double_column(imags);
  for (int64_t i = 0; i < n; ++i) out[i] = rcol[i] + sign[i] + icol[i] + "j";
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
