#include "numpp/io/npy.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <string>

namespace numpp {
namespace {

struct DescrInfo { DType dt; const char* descr; };
// itemsize 1 -> '|' byte-order, else '<' (native little-endian).
const DescrInfo kDescrs[] = {
    {kBool, "|b1"}, {kInt8, "|i1"}, {kInt16, "<i2"}, {kInt32, "<i4"}, {kInt64, "<i8"},
    {kUInt8, "|u1"}, {kUInt16, "<u2"}, {kUInt32, "<u4"}, {kUInt64, "<u8"},
    {kFloat16, "<f2"}, {kFloat32, "<f4"}, {kFloat64, "<f8"},
    {kComplex64, "<c8"}, {kComplex128, "<c16"},
};

std::string shape_tuple(const Shape& s) {
  std::string out = "(";
  for (size_t i = 0; i < s.size(); ++i) { out += std::to_string(s[i]); out += ", "; }
  if (s.size() == 1) out.pop_back();  // "(3, )" -> "(3,)" : drop the space
  else if (!s.empty()) { out.pop_back(); out.pop_back(); }  // drop trailing ", "
  out += ")";
  return out;
}

}  // namespace

std::string dtype_to_descr(DType d) {
  if (d.kind() == 'U') return "<U" + std::to_string(d.itemsize() / 4);
  if (d.kind() == 'S') return "|S" + std::to_string(d.itemsize());
  for (const auto& e : kDescrs) if (e.dt == d) return e.descr;
  throw type_error("npy: unsupported dtype");
}
DType descr_to_dtype(const std::string& descr) {
  std::string suffix = descr.empty() ? descr : descr.substr(1);  // drop byte-order char
  if (!suffix.empty() && (suffix[0] == 'U' || suffix[0] == 'S')) {
    int64_t n = std::stoll(suffix.substr(1));
    return suffix[0] == 'U' ? make_string(n) : make_bytes(n);
  }
  for (const auto& e : kDescrs) if (std::string(e.descr).substr(1) == suffix) return e.dt;
  throw type_error("npy: unsupported descr '" + descr + "'");
}

std::string npy_bytes(const ndarray& a) {
  ndarray c = a.ascontiguousarray();
  std::string header = "{'descr': '" + dtype_to_descr(c.dtype()) +
                       "', 'fortran_order': False, 'shape': " + shape_tuple(c.shape()) + ", }";
  // pad so that (10 + header.size() + 1) % 64 == 0, header ends with '\n'
  size_t total = 10 + header.size() + 1;
  size_t pad = (64 - (total % 64)) % 64;
  header.append(pad, ' ');
  header.push_back('\n');
  std::string out;
  out.append("\x93NUMPY", 6);
  out.push_back('\x01'); out.push_back('\x00');           // version 1.0
  uint16_t hlen = static_cast<uint16_t>(header.size());
  out.push_back(static_cast<char>(hlen & 0xff));
  out.push_back(static_cast<char>((hlen >> 8) & 0xff));
  out += header;
  out.append(c.bytes(), static_cast<size_t>(c.nbytes()));
  return out;
}

ndarray npy_from_bytes(const char* data, size_t size) {
  if (size < 10 || std::memcmp(data, "\x93NUMPY", 6) != 0) throw value_error("npy: bad magic");
  const uint8_t major = static_cast<uint8_t>(data[6]);
  size_t hlen, hoff;
  if (major == 1) {
    hlen = static_cast<uint8_t>(data[8]) | (static_cast<uint8_t>(data[9]) << 8);
    hoff = 10;
  } else {  // v2.0+: 4-byte header length
    hlen = static_cast<uint8_t>(data[8]) | (static_cast<uint8_t>(data[9]) << 8) |
           (static_cast<uint8_t>(data[10]) << 16) | (static_cast<uint32_t>(static_cast<uint8_t>(data[11])) << 24);
    hoff = 12;
  }
  std::string header(data + hoff, hlen);
  auto field = [&](const std::string& key) -> std::string {
    size_t p = header.find(key);
    return p == std::string::npos ? std::string() : header.substr(p + key.size());
  };
  std::string ds = field("'descr':");
  size_t q1 = ds.find('\''), q2 = ds.find('\'', q1 + 1);
  DType dt = descr_to_dtype(ds.substr(q1 + 1, q2 - q1 - 1));
  std::string fo = field("'fortran_order':");
  bool fortran = fo.substr(0, fo.find(',')).find("True") != std::string::npos;
  std::string ss = field("'shape':");
  size_t lp = ss.find('('), rp = ss.find(')');
  Shape shape;
  std::string inside = ss.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < inside.size();) {
    if (std::isdigit(static_cast<unsigned char>(inside[i]))) {
      int64_t v = 0;
      while (i < inside.size() && std::isdigit(static_cast<unsigned char>(inside[i]))) v = v * 10 + (inside[i++] - '0');
      shape.push_back(v);
    } else ++i;
  }
  ndarray out(shape, dt, fortran ? Order::F : Order::C);
  std::memcpy(out.bytes(), data + hoff + hlen, static_cast<size_t>(out.nbytes()));
  return out;
}

void save(const std::string& path, const ndarray& a) {
  std::string bytes = npy_bytes(a);
  std::ofstream f(path, std::ios::binary);
  if (!f) throw value_error("npy: cannot open '" + path + "' for writing");
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
ndarray load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw value_error("npy: cannot open '" + path + "'");
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return npy_from_bytes(buf.data(), buf.size());
}

}  // namespace numpp
