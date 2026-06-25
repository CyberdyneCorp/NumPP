#include "numpp/struct/struct.hpp"

namespace numpp {

DType make_struct(const std::vector<std::pair<std::string, DType>>& fields) {
  auto m = std::make_shared<DTypeMeta>();
  int64_t offset = 0;
  for (const auto& [name, dt] : fields) {
    m->fields.push_back(StructField{name, dt, offset});
    offset += dt.itemsize();
  }
  m->itemsize = offset;
  return DType(DTypeId::Struct, m);
}

ndarray field_view(const ndarray& a, const std::string& name) {
  if (a.dtype().kind() != 'V' || !a.dtype().meta()) throw type_error("field_view: not a structured array");
  for (const auto& f : a.dtype().meta()->fields) {
    if (f.name == name)
      return ndarray(a.buffer(), f.dtype, a.shape(), a.strides(), a.offset() + f.offset, a.writeable());
  }
  throw value_error("field_view: no field named '" + name + "'");
}

}  // namespace numpp
