#include "numpp/core/dtype.hpp"

#include <array>
#include <cstdint>
#include <unordered_map>

namespace numpp {
namespace {

constexpr int N = kNumDTypes;

// Tables generated from numpy 2.1.3 (see openspec design notes). Index order
// matches DTypeId. These are the source of truth for promotion/casting and are
// verified element-by-element against numpy in the oracle test suite.
constexpr uint8_t kPromote[N][N] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13},
  {1,1,2,3,4,2,3,4,11,9,10,11,12,13},
  {2,2,2,3,4,2,3,4,11,10,10,11,12,13},
  {3,3,3,3,4,3,3,4,11,11,11,11,13,13},
  {4,4,4,4,4,4,4,4,11,11,11,11,13,13},
  {5,2,2,3,4,5,6,7,8,9,10,11,12,13},
  {6,3,3,3,4,6,6,7,8,10,10,11,12,13},
  {7,4,4,4,4,7,7,7,8,11,11,11,13,13},
  {8,11,11,11,11,8,8,8,8,11,11,11,13,13},
  {9,9,10,11,11,9,10,11,11,9,10,11,12,13},
  {10,10,10,11,11,10,10,11,11,10,10,11,12,13},
  {11,11,11,11,11,11,11,11,11,11,11,11,13,13},
  {12,12,12,13,13,12,12,13,13,12,12,13,12,13},
  {13,13,13,13,13,13,13,13,13,13,13,13,13,13},
};
constexpr uint8_t kCanCastSafe[N][N] = {
  {1,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,1,0,0,0,0,1,1,1,1,1},
  {0,0,1,1,1,0,0,0,0,0,1,1,1,1},
  {0,0,0,1,1,0,0,0,0,0,0,1,0,1},
  {0,0,0,0,1,0,0,0,0,0,0,1,0,1},
  {0,0,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,0,0,1,1,0,1,1,1,0,1,1,1,1},
  {0,0,0,0,1,0,0,1,1,0,0,1,0,1},
  {0,0,0,0,0,0,0,0,1,0,0,1,0,1},
  {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,0,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,0,0,1,0,1},
  {0,0,0,0,0,0,0,0,0,0,0,0,1,1},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,1},
};
constexpr uint8_t kCanCastSameKind[N][N] = {
  {1,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,1,0,0,0,0,1,1,1,1,1},
  {0,1,1,1,1,0,0,0,0,1,1,1,1,1},
  {0,1,1,1,1,0,0,0,0,1,1,1,1,1},
  {0,1,1,1,1,0,0,0,0,1,1,1,1,1},
  {0,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,1,1,1,1,1,1,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
  {0,0,0,0,0,0,0,0,0,0,0,0,1,1},
  {0,0,0,0,0,0,0,0,0,0,0,0,1,1},
};

struct Meta { uint8_t size; char kind; const char* name; };
constexpr std::array<Meta, N> kMeta = {{
  {1,'b',"bool"}, {1,'i',"int8"}, {2,'i',"int16"}, {4,'i',"int32"}, {8,'i',"int64"},
  {1,'u',"uint8"}, {2,'u',"uint16"}, {4,'u',"uint32"}, {8,'u',"uint64"},
  {2,'f',"float16"}, {4,'f',"float32"}, {8,'f',"float64"},
  {8,'c',"complex64"}, {16,'c',"complex128"},
}};

int ix(DType d) { return static_cast<int>(d.id()); }

}  // namespace

uint8_t DType::itemsize() const { return kMeta[static_cast<int>(id_)].size; }
char DType::kind() const { return kMeta[static_cast<int>(id_)].kind; }
const char* DType::name() const { return kMeta[static_cast<int>(id_)].name; }
bool DType::is_floating() const { return kind() == 'f'; }
bool DType::is_integer() const { return kind() == 'i' || kind() == 'u'; }

DType default_int() {
  return sizeof(void*) >= 8 ? kInt64 : kInt32;
}

DType DType::from_name(std::string_view s) {
  static const std::unordered_map<std::string_view, DTypeId> kNames = {
    {"bool", DTypeId::Bool}, {"b1", DTypeId::Bool},
    {"int8", DTypeId::Int8}, {"i1", DTypeId::Int8},
    {"int16", DTypeId::Int16}, {"i2", DTypeId::Int16},
    {"int32", DTypeId::Int32}, {"i4", DTypeId::Int32},
    {"int64", DTypeId::Int64}, {"i8", DTypeId::Int64},
    {"uint8", DTypeId::UInt8}, {"u1", DTypeId::UInt8},
    {"uint16", DTypeId::UInt16}, {"u2", DTypeId::UInt16},
    {"uint32", DTypeId::UInt32}, {"u4", DTypeId::UInt32},
    {"uint64", DTypeId::UInt64}, {"u8", DTypeId::UInt64},
    {"float16", DTypeId::Float16}, {"f2", DTypeId::Float16}, {"half", DTypeId::Float16},
    {"float32", DTypeId::Float32}, {"f4", DTypeId::Float32}, {"single", DTypeId::Float32},
    {"float64", DTypeId::Float64}, {"f8", DTypeId::Float64}, {"double", DTypeId::Float64},
    {"complex64", DTypeId::Complex64}, {"c8", DTypeId::Complex64},
    {"complex128", DTypeId::Complex128}, {"c16", DTypeId::Complex128},
  };
  if (auto it = kNames.find(s); it != kNames.end()) return DType(it->second);
  // Aliases that map to platform defaults.
  if (s == "float") return kDefaultFloat;
  if (s == "complex") return kDefaultComplex;
  if (s == "int") return default_int();
  throw type_error("unknown dtype name: " + std::string(s));
}

DType result_type(DType a, DType b) {
  return DType(static_cast<DTypeId>(kPromote[ix(a)][ix(b)]));
}

bool can_cast(DType from, DType to, Casting casting) {
  switch (casting) {
    case Casting::No:
    case Casting::Equiv:   return from == to;
    case Casting::Safe:    return kCanCastSafe[ix(from)][ix(to)] != 0;
    case Casting::SameKind:return kCanCastSameKind[ix(from)][ix(to)] != 0;
    case Casting::Unsafe:  return true;
  }
  return false;
}

}  // namespace numpp
