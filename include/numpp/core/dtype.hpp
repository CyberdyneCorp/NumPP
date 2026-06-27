#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "numpp/core/error.hpp"
#include "numpp/core/half.hpp"
#include "numpp/export.hpp"

namespace numpp {

enum class DTypeId : uint8_t {
  Bool = 0,
  Int8, Int16, Int32, Int64,
  UInt8, UInt16, UInt32, UInt64,
  Float16, Float32, Float64,
  Complex64, Complex128,
  // Extended dtypes (carry a metadata side-channel; not in the numeric tables).
  String, Bytes, Datetime64, Timedelta64, Struct,
};

inline constexpr int kNumDTypes = 14;  // count of numeric dtypes (table dimension)

struct DTypeMeta;  // defined below (needs complete DType)

// A dtype value: a numeric id, or an extended id plus shared metadata. Numeric
// dtypes keep a null metadata pointer, so they stay cheap to copy/compare.
class NUMPP_API DType {
 public:
  DType() = default;
  explicit DType(DTypeId id) : id_(id) {}
  DType(DTypeId id, std::shared_ptr<const DTypeMeta> meta) : id_(id), meta_(std::move(meta)) {}

  static DType from_name(std::string_view name);

  DTypeId id() const { return id_; }
  int64_t itemsize() const;
  char kind() const;          // 'b','i','u','f','c','U','S','M','m','V'
  const char* name() const;   // "int32", "float64", ...
  char byteorder() const { return '='; }  // native only
  const std::shared_ptr<const DTypeMeta>& meta() const { return meta_; }

  bool is_complex() const { return id_ == DTypeId::Complex64 || id_ == DTypeId::Complex128; }
  bool is_floating() const;
  bool is_integer() const;
  bool is_numeric() const { return static_cast<int>(id_) < kNumDTypes; }
  bool is_extended() const { return static_cast<int>(id_) >= kNumDTypes; }

  NUMPP_API friend bool operator==(const DType& a, const DType& b);
  friend bool operator!=(const DType& a, const DType& b) { return !(a == b); }

 private:
  DTypeId id_ = DTypeId::Float64;
  std::shared_ptr<const DTypeMeta> meta_;
};

struct StructField { std::string name; DType dtype; int64_t offset; };
// Side-channel metadata for extended dtypes; null for the 14 numeric dtypes.
struct DTypeMeta {
  int64_t itemsize = 0;          // string/bytes/datetime/struct byte size
  char unit = '\0';              // datetime64/timedelta64 unit (Y/M/D/h/m/s/...)
  std::vector<StructField> fields;  // structured dtype layout
};

// Extended-dtype factories.
NUMPP_API DType make_string(int64_t num_chars);  // 'U' (UTF-32, 4 bytes/char)
NUMPP_API DType make_bytes(int64_t num_bytes);   // 'S'

// Convenience dtype constants.
inline const DType kBool{DTypeId::Bool};
inline const DType kInt8{DTypeId::Int8};
inline const DType kInt16{DTypeId::Int16};
inline const DType kInt32{DTypeId::Int32};
inline const DType kInt64{DTypeId::Int64};
inline const DType kUInt8{DTypeId::UInt8};
inline const DType kUInt16{DTypeId::UInt16};
inline const DType kUInt32{DTypeId::UInt32};
inline const DType kUInt64{DTypeId::UInt64};
inline const DType kFloat16{DTypeId::Float16};
inline const DType kFloat32{DTypeId::Float32};
inline const DType kFloat64{DTypeId::Float64};
inline const DType kComplex64{DTypeId::Complex64};
inline const DType kComplex128{DTypeId::Complex128};

// Default dtypes (NumPy parity on LP64 platforms).
inline const DType kDefaultFloat = kFloat64;
inline const DType kDefaultComplex = kComplex128;
NUMPP_API DType default_int();  // typically int64 on 64-bit platforms

enum class Casting { No, Equiv, Safe, SameKind, Unsafe };

// NEP 50 dtype-based promotion. Commutative, associative; matches numpy.result_type.
NUMPP_API DType result_type(DType a, DType b);

// Matches numpy.can_cast for the supported casting modes.
NUMPP_API bool can_cast(DType from, DType to, Casting casting);

// numpy.promote_types: the smallest dtype both inputs can be safely cast to
// (purely type-based, no value inspection). Numeric dtypes only.
NUMPP_API DType promote_types(DType a, DType b);

// numpy.min_scalar_type: the smallest dtype that can represent a scalar value.
// Non-negative integers map to the smallest unsigned type, negative integers to
// the smallest signed type, and floating values to the smallest float whose range
// holds the magnitude (float16 -> float32 -> float64).
NUMPP_API DType min_scalar_type(long long value);
NUMPP_API DType min_scalar_type(unsigned long long value);
NUMPP_API DType min_scalar_type(double value);

// Maps a DTypeId to its concrete C++ storage type for generic kernels.
template <DTypeId Id> struct dtype_traits;
template <> struct dtype_traits<DTypeId::Bool>       { using type = bool; };
template <> struct dtype_traits<DTypeId::Int8>       { using type = int8_t; };
template <> struct dtype_traits<DTypeId::Int16>      { using type = int16_t; };
template <> struct dtype_traits<DTypeId::Int32>      { using type = int32_t; };
template <> struct dtype_traits<DTypeId::Int64>      { using type = int64_t; };
template <> struct dtype_traits<DTypeId::UInt8>      { using type = uint8_t; };
template <> struct dtype_traits<DTypeId::UInt16>     { using type = uint16_t; };
template <> struct dtype_traits<DTypeId::UInt32>     { using type = uint32_t; };
template <> struct dtype_traits<DTypeId::UInt64>     { using type = uint64_t; };
template <> struct dtype_traits<DTypeId::Float16>    { using type = half; };
template <> struct dtype_traits<DTypeId::Float32>    { using type = float; };
template <> struct dtype_traits<DTypeId::Float64>    { using type = double; };
template <> struct dtype_traits<DTypeId::Complex64>  { using type = std::complex<float>; };
template <> struct dtype_traits<DTypeId::Complex128> { using type = std::complex<double>; };

// Dispatch a DTypeId to a generic functor: visit(id, [](auto tag){ using T = ...; }).
// Keeps per-dtype branching in one place (low cognitive complexity at call sites).
template <class F>
decltype(auto) visit_dtype(DTypeId id, F&& f) {
  switch (id) {
    case DTypeId::Bool:       return f(dtype_traits<DTypeId::Bool>{});
    case DTypeId::Int8:       return f(dtype_traits<DTypeId::Int8>{});
    case DTypeId::Int16:      return f(dtype_traits<DTypeId::Int16>{});
    case DTypeId::Int32:      return f(dtype_traits<DTypeId::Int32>{});
    case DTypeId::Int64:      return f(dtype_traits<DTypeId::Int64>{});
    case DTypeId::UInt8:      return f(dtype_traits<DTypeId::UInt8>{});
    case DTypeId::UInt16:     return f(dtype_traits<DTypeId::UInt16>{});
    case DTypeId::UInt32:     return f(dtype_traits<DTypeId::UInt32>{});
    case DTypeId::UInt64:     return f(dtype_traits<DTypeId::UInt64>{});
    case DTypeId::Float16:    return f(dtype_traits<DTypeId::Float16>{});
    case DTypeId::Float32:    return f(dtype_traits<DTypeId::Float32>{});
    case DTypeId::Float64:    return f(dtype_traits<DTypeId::Float64>{});
    case DTypeId::Complex64:  return f(dtype_traits<DTypeId::Complex64>{});
    case DTypeId::Complex128: return f(dtype_traits<DTypeId::Complex128>{});
    case DTypeId::String: case DTypeId::Bytes: case DTypeId::Datetime64:
    case DTypeId::Timedelta64: case DTypeId::Struct:
      throw type_error("visit_dtype: not a numeric dtype");
  }
  throw type_error("invalid dtype id");
}

}  // namespace numpp
