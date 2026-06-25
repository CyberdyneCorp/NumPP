#pragma once

#include <string>
#include <utility>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Structured dtype with numpy default packing (offset = running sum of field
// itemsizes, no alignment padding).
NUMPP_API DType make_struct(const std::vector<std::pair<std::string, DType>>& fields);

// View of one field across all records: shares the parent buffer, the field's
// dtype, the parent's strides (which already step by the record size), and an
// offset shifted by the field's offset — so reads/writes alias the parent.
NUMPP_API ndarray field_view(const ndarray& a, const std::string& name);

template <class T>
T get_field(const ndarray& a, int64_t record, const std::string& name) {
  return field_view(a, name).template item<T>({record});
}
template <class T>
void set_field(ndarray& a, int64_t record, const std::string& name, T value) {
  ndarray fv = field_view(a, name);
  fv.set_item<T>({record}, value);
}

}  // namespace numpp
