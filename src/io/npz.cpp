#include "numpp/io/npz.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>

#include "numpp/io/npy.hpp"

namespace numpp {
namespace {

uint32_t crc32(const char* data, size_t n) {
  static std::array<uint32_t, 256> table;
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) c = table[(c ^ static_cast<uint8_t>(data[i])) & 0xff] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

void put16(std::string& s, uint16_t v) { s.push_back(char(v & 0xff)); s.push_back(char((v >> 8) & 0xff)); }
void put32(std::string& s, uint32_t v) { for (int i = 0; i < 4; ++i) s.push_back(char((v >> (8 * i)) & 0xff)); }
uint16_t get16(const char* p) { return uint8_t(p[0]) | (uint8_t(p[1]) << 8); }
uint32_t get32(const char* p) { return uint8_t(p[0]) | (uint8_t(p[1]) << 8) | (uint8_t(p[2]) << 16) | (uint32_t(uint8_t(p[3])) << 24); }

void write_zip(const std::string& path, const std::vector<NamedArray>& arrays) {
  std::string out;
  struct CD { std::string name; uint32_t crc, size, offset; };
  std::vector<CD> cds;
  for (const auto& [name, arr] : arrays) {
    std::string npy = npy_bytes(arr);
    std::string fname = name + ".npy";
    uint32_t crc = crc32(npy.data(), npy.size());
    uint32_t offset = static_cast<uint32_t>(out.size());
    put32(out, 0x04034b50); put16(out, 20); put16(out, 0); put16(out, 0);  // sig, ver, flags, method=STORED
    put16(out, 0); put16(out, 0);                                          // mod time, date
    put32(out, crc); put32(out, uint32_t(npy.size())); put32(out, uint32_t(npy.size()));
    put16(out, uint16_t(fname.size())); put16(out, 0);                     // name len, extra len
    out += fname;
    out += npy;
    cds.push_back({fname, crc, uint32_t(npy.size()), offset});
  }
  uint32_t cdstart = static_cast<uint32_t>(out.size());
  for (const auto& cd : cds) {
    put32(out, 0x02014b50); put16(out, 20); put16(out, 20); put16(out, 0); put16(out, 0);  // sig, made, need, flags, method
    put16(out, 0); put16(out, 0);                                                          // time, date
    put32(out, cd.crc); put32(out, cd.size); put32(out, cd.size);
    put16(out, uint16_t(cd.name.size())); put16(out, 0); put16(out, 0);                    // name, extra, comment
    put16(out, 0); put16(out, 0); put32(out, 0);                                           // disk, int attr, ext attr
    put32(out, cd.offset);
    out += cd.name;
  }
  uint32_t cdsize = static_cast<uint32_t>(out.size()) - cdstart;
  put32(out, 0x06054b50); put16(out, 0); put16(out, 0);
  put16(out, uint16_t(cds.size())); put16(out, uint16_t(cds.size()));
  put32(out, cdsize); put32(out, cdstart); put16(out, 0);
  std::ofstream f(path, std::ios::binary);
  if (!f) throw value_error("npz: cannot open '" + path + "' for writing");
  f.write(out.data(), static_cast<std::streamsize>(out.size()));
}

}  // namespace

void savez(const std::string& path, const std::vector<NamedArray>& arrays) { write_zip(path, arrays); }
void savez_compressed(const std::string& path, const std::vector<NamedArray>& arrays) { write_zip(path, arrays); }

std::map<std::string, ndarray> load_npz(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw value_error("npz: cannot open '" + path + "'");
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const char* d = buf.data();
  const size_t n = buf.size();
  // Find End Of Central Directory (scan back for the signature).
  size_t eocd = std::string::npos;
  for (size_t i = n >= 22 ? n - 22 : 0; ; --i) { if (get32(d + i) == 0x06054b50u) { eocd = i; break; } if (i == 0) break; }
  if (eocd == std::string::npos) throw value_error("npz: not a zip file");
  uint16_t entries = get16(d + eocd + 10);
  uint32_t cdoff = get32(d + eocd + 16);
  std::map<std::string, ndarray> out;
  size_t p = cdoff;
  for (uint16_t e = 0; e < entries; ++e) {
    if (get32(d + p) != 0x02014b50u) break;
    uint16_t nameLen = get16(d + p + 28), extraLen = get16(d + p + 30), commentLen = get16(d + p + 32);
    uint32_t usize = get32(d + p + 24);
    uint32_t localoff = get32(d + p + 42);
    std::string name(d + p + 46, nameLen);
    // Local header at localoff: data starts after fixed 30 bytes + local name/extra.
    uint16_t lnameLen = get16(d + localoff + 26), lextraLen = get16(d + localoff + 28);
    const char* data = d + localoff + 30 + lnameLen + lextraLen;
    std::string key = name.size() > 4 && name.substr(name.size() - 4) == ".npy" ? name.substr(0, name.size() - 4) : name;
    out.emplace(key, npy_from_bytes(data, usize));
    p += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

}  // namespace numpp
