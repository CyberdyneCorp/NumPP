#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/error.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Reference-counted raw data buffer. Multiple arrays/views share one Buffer;
// it is freed when the last referencing array is destroyed (RAII).
class NUMPP_API Buffer {
 public:
  explicit Buffer(int64_t nbytes) : data_(static_cast<char*>(::operator new(static_cast<size_t>(nbytes)))),
                                    nbytes_(nbytes) {}
  // Adopt external memory without owning it (no-copy construction).
  Buffer(char* external, int64_t nbytes, bool owns)
      : data_(external), nbytes_(nbytes), owns_(owns) {}
  ~Buffer() { if (owns_) ::operator delete(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  char* data() { return data_; }
  const char* data() const { return data_; }
  int64_t nbytes() const { return nbytes_; }

 private:
  char* data_ = nullptr;
  int64_t nbytes_ = 0;
  bool owns_ = true;
};

// A slice with optional bounds, like Python's start:stop:step.
struct Slice {
  std::optional<int64_t> start, stop;
  int64_t step = 1;
};
struct NewAxis {};
using IndexItem = std::variant<int64_t, Slice, NewAxis>;

// Map a C++ scalar type to its dtype.
template <class T> DType dtype_of();

class NUMPP_API ndarray {
 public:
  ndarray() = default;

  // Owning array with C-contiguous (or F) storage of the given shape+dtype.
  ndarray(Shape shape, DType dtype, Order order = Order::C);

  // View over an existing buffer (no copy).
  ndarray(std::shared_ptr<Buffer> buffer, DType dtype, Shape shape, Strides strides,
          int64_t offset, bool writeable);

  // --- metadata ---
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
  int64_t size() const { return shape_size(shape_); }
  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  int64_t itemsize() const { return dtype_.itemsize(); }
  int64_t nbytes() const { return size() * itemsize(); }
  int64_t offset() const { return offset_; }
  const std::shared_ptr<Buffer>& buffer() const { return buffer_; }

  bool c_contiguous() const { return is_c_contiguous(shape_, strides_, itemsize()); }
  bool f_contiguous() const { return is_f_contiguous(shape_, strides_, itemsize()); }
  bool writeable() const { return writeable_; }

  // --- raw data access ---
  char* bytes() { return buffer_->data() + offset_; }
  const char* bytes() const { return buffer_->data() + offset_; }
  char* element_ptr(const std::vector<int64_t>& idx);
  const char* element_ptr(const std::vector<int64_t>& idx) const;

  template <class T> T* typed_data() {
    require_dtype<T>();
    require_writeable();
    return reinterpret_cast<T*>(bytes());
  }
  template <class T> const T* typed_data() const {
    require_dtype<T>();
    return reinterpret_cast<const T*>(bytes());
  }

  // Read one element (dtype must match T).
  template <class T> T item(const std::vector<int64_t>& idx) const {
    require_dtype<T>();
    T v;
    std::memcpy(&v, element_ptr(idx), sizeof(T));
    return v;
  }
  template <class T> void set_item(const std::vector<int64_t>& idx, T v) {
    require_dtype<T>();
    require_writeable();
    std::memcpy(element_ptr(idx), &v, sizeof(T));
  }
  template <class T> void fill(T v) {
    require_dtype<T>();
    require_writeable();
    fill_bytes(&v);
  }

  // --- shape ops (views unless noted) ---
  ndarray reshape(const Shape& new_shape) const;
  ndarray transpose() const;
  ndarray transpose(const std::vector<int64_t>& axes) const;
  ndarray swapaxes(int64_t a, int64_t b) const;
  ndarray squeeze() const;
  ndarray expand_dims(int64_t axis) const;
  ndarray ravel(Order order = Order::C) const;   // view if possible, else copy
  ndarray flatten(Order order = Order::C) const;  // always a copy

  // --- indexing (basic; returns a view) ---
  ndarray index(const std::vector<IndexItem>& items) const;
  ndarray operator[](int64_t i) const;  // integer index on axis 0

  // --- copies / layout ---
  ndarray copy(Order order = Order::C) const;
  ndarray ascontiguousarray() const;
  ndarray asfortranarray() const;
  ndarray broadcast_to(const Shape& shape) const;
  ndarray astype(DType dt, Casting casting = Casting::Unsafe) const;

  // Visit every element offset in C order (handles non-contiguous / zero-stride).
  template <class F>
  void for_each_offset(F&& f) const {
    const int64_t n = ndim();
    if (size() == 0) return;
    std::vector<int64_t> idx(n, 0);
    while (true) {
      int64_t off = offset_;
      for (int64_t i = 0; i < n; ++i) off += idx[i] * strides_[i];
      f(off);
      int64_t ax = n - 1;
      for (; ax >= 0; --ax) {
        if (++idx[ax] < shape_[ax]) break;
        idx[ax] = 0;
      }
      if (ax < 0) break;
    }
  }

 private:
  template <class T> void require_dtype() const {
    if (dtype_ != dtype_of<T>()) throw type_error("array dtype does not match requested type");
  }
  void require_writeable() const {
    if (!writeable_) throw value_error("assignment destination is read-only");
  }
  void fill_bytes(const void* scalar);

  std::shared_ptr<Buffer> buffer_;
  DType dtype_{DTypeId::Float64};
  Shape shape_;
  Strides strides_;
  int64_t offset_ = 0;
  bool writeable_ = true;
};

// --- comparison helpers ---
NUMPP_API bool array_equal(const ndarray& a, const ndarray& b);
NUMPP_API bool allclose(const ndarray& a, const ndarray& b, double rtol = 1e-5,
                        double atol = 1e-8, bool equal_nan = false);

// dtype_of specializations
template <> inline DType dtype_of<bool>() { return kBool; }
template <> inline DType dtype_of<int8_t>() { return kInt8; }
template <> inline DType dtype_of<int16_t>() { return kInt16; }
template <> inline DType dtype_of<int32_t>() { return kInt32; }
template <> inline DType dtype_of<int64_t>() { return kInt64; }
template <> inline DType dtype_of<uint8_t>() { return kUInt8; }
template <> inline DType dtype_of<uint16_t>() { return kUInt16; }
template <> inline DType dtype_of<uint32_t>() { return kUInt32; }
template <> inline DType dtype_of<uint64_t>() { return kUInt64; }
template <> inline DType dtype_of<half>() { return kFloat16; }
template <> inline DType dtype_of<float>() { return kFloat32; }
template <> inline DType dtype_of<double>() { return kFloat64; }
template <> inline DType dtype_of<std::complex<float>>() { return kComplex64; }
template <> inline DType dtype_of<std::complex<double>>() { return kComplex128; }

}  // namespace numpp
