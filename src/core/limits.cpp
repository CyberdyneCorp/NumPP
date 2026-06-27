#include "numpp/core/limits.hpp"

#include <cmath>
#include <limits>

#include "numpp/core/error.hpp"

namespace numpp {
namespace {

// Build FInfo for a concrete IEEE real type from std::numeric_limits.
template <class T>
FInfo finfo_of() {
  FInfo f;
  f.nmant = std::numeric_limits<T>::digits - 1;            // explicit mantissa bits
  f.bits = static_cast<int>(sizeof(T) * 8);
  f.nexp = f.bits - f.nmant - 1;
  f.precision = std::numeric_limits<T>::digits10;
  f.eps = static_cast<double>(std::numeric_limits<T>::epsilon());
  f.epsneg = f.eps / 2.0;
  f.tiny = static_cast<double>(std::numeric_limits<T>::min());      // smallest normal
  f.smallest_normal = f.tiny;
  f.smallest_subnormal = static_cast<double>(std::numeric_limits<T>::denorm_min());
  f.max = static_cast<double>(std::numeric_limits<T>::max());
  f.min = -f.max;
  f.resolution = std::pow(10.0, -f.precision);
  return f;
}

// float16 is not a std::numeric_limits type; use numpy's IEEE-half constants.
FInfo finfo_half() {
  FInfo f;
  f.bits = 16;
  f.nmant = 10;
  f.nexp = 5;
  f.precision = 3;
  f.eps = 0.0009765625;                 // 2**-10
  f.epsneg = 0.00048828125;             // 2**-11
  f.tiny = 6.103515625e-05;             // 2**-14 smallest normal
  f.smallest_normal = f.tiny;
  f.smallest_subnormal = 5.9604644775390625e-08;  // 2**-24
  f.max = 65504.0;
  f.min = -65504.0;
  f.resolution = 0.001;                 // 10**-3
  return f;
}

template <class T>
IInfo iinfo_of() {
  IInfo i;
  i.bits = static_cast<int>(sizeof(T) * 8);
  i.min = static_cast<long long>(std::numeric_limits<T>::min());
  i.max = static_cast<unsigned long long>(std::numeric_limits<T>::max());
  return i;
}

}  // namespace

FInfo finfo(DType dt) {
  switch (dt.id()) {
    case DTypeId::Float16: return finfo_half();
    case DTypeId::Float32:
    case DTypeId::Complex64: return finfo_of<float>();
    case DTypeId::Float64:
    case DTypeId::Complex128: return finfo_of<double>();
    default:
      throw value_error("finfo: dtype is not floating or complex");
  }
}

IInfo iinfo(DType dt) {
  switch (dt.id()) {
    case DTypeId::Bool: { IInfo i{1, 0, 1}; return i; }
    case DTypeId::Int8: return iinfo_of<int8_t>();
    case DTypeId::Int16: return iinfo_of<int16_t>();
    case DTypeId::Int32: return iinfo_of<int32_t>();
    case DTypeId::Int64: return iinfo_of<int64_t>();
    case DTypeId::UInt8: return iinfo_of<uint8_t>();
    case DTypeId::UInt16: return iinfo_of<uint16_t>();
    case DTypeId::UInt32: return iinfo_of<uint32_t>();
    case DTypeId::UInt64: return iinfo_of<uint64_t>();
    default:
      throw value_error("iinfo: dtype is not an integer type");
  }
}

}  // namespace numpp
