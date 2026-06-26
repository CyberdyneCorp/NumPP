#include "numpp/interop/interop.hpp"

#include "numpp/core/ndarray.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/error.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace numpp {

namespace interop {

// Non-capturing free function used as the DLPack deleter (C function pointer).
void numpp_dlpack_deleter(DLManagedTensor* self) {
    if (self == nullptr) return;
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<ndarray*>(self->manager_ctx);
    delete self;
}

DLManagedTensor* to_dlpack(const ndarray& a) {
    ndarray ac = a.ascontiguousarray();
    const int32_t ndim = static_cast<int32_t>(ac.ndim());
    const int64_t itemsize = ac.itemsize();

    DLManagedTensor* t = new DLManagedTensor();
    t->dl_tensor.data = ac.bytes();
    t->dl_tensor.device = DLDevice{kDLCPU, 0};
    t->dl_tensor.ndim = ndim;
    t->dl_tensor.byte_offset = 0;

    const char kind = ac.dtype().kind();
    uint8_t code;
    switch (kind) {
        case 'i': code = kDLInt; break;
        case 'u': code = kDLUInt; break;
        case 'f': code = kDLFloat; break;
        case 'c': code = kDLComplex; break;
        case 'b': code = kDLBool; break;
        default:  code = kDLUInt; break;
    }
    t->dl_tensor.dtype = DLDataType{code, static_cast<uint8_t>(itemsize * 8), 1};

    int64_t* shape = new int64_t[ndim];
    int64_t* strides = new int64_t[ndim];
    const Shape& sh = ac.shape();
    const Strides& st = ac.strides();
    for (int32_t i = 0; i < ndim; ++i) {
        shape[i] = sh[i];
        strides[i] = st[i] / itemsize;  // ELEMENT strides
    }
    t->dl_tensor.shape = shape;
    t->dl_tensor.strides = strides;

    // Keep the underlying buffer alive for the lifetime of the managed tensor.
    t->manager_ctx = new ndarray(ac);
    t->deleter = &numpp_dlpack_deleter;
    return t;
}

ndarray from_dlpack(DLManagedTensor* t) {
    const DLTensor& dt = t->dl_tensor;
    const uint8_t code = dt.dtype.code;
    const uint8_t bits = dt.dtype.bits;
    const int64_t itemsize = bits / 8;

    DType dtype = kUInt8;
    if (code == kDLInt) {
        switch (bits) {
            case 8:  dtype = kInt8; break;
            case 16: dtype = kInt16; break;
            case 32: dtype = kInt32; break;
            case 64: dtype = kInt64; break;
            default: throw value_error("from_dlpack: unsupported int bits");
        }
    } else if (code == kDLUInt) {
        switch (bits) {
            case 8:  dtype = kUInt8; break;
            case 16: dtype = kUInt16; break;
            case 32: dtype = kUInt32; break;
            case 64: dtype = kUInt64; break;
            default: throw value_error("from_dlpack: unsupported uint bits");
        }
    } else if (code == kDLFloat) {
        switch (bits) {
            case 16: dtype = kFloat16; break;
            case 32: dtype = kFloat32; break;
            case 64: dtype = kFloat64; break;
            default: throw value_error("from_dlpack: unsupported float bits");
        }
    } else if (code == kDLComplex) {
        dtype = (bits == 128) ? kComplex128 : kComplex64;
    } else if (code == kDLBool) {
        dtype = kBool;
    } else {
        throw value_error("from_dlpack: unsupported DLDataType code");
    }

    char* ptr = static_cast<char*>(dt.data) + dt.byte_offset;
    Shape shape(dt.shape, dt.shape + dt.ndim);

    Strides strides;
    if (dt.strides == nullptr) {
        strides = contiguous_strides(shape, itemsize, Order::C);
    } else {
        for (int32_t i = 0; i < dt.ndim; ++i) {
            strides.push_back(dt.strides[i] * itemsize);  // element -> byte strides
        }
    }

    int64_t nelem = 1;
    for (int32_t i = 0; i < dt.ndim; ++i) nelem *= dt.shape[i];
    const int64_t nbytes = nelem * itemsize;

    std::shared_ptr<Buffer> buf(new Buffer(ptr, nbytes, /*owns=*/false),
        [t](Buffer* b) {
            delete b;
            if (t->deleter) t->deleter(t);
        });

    return ndarray(buf, dtype, shape, strides, 0, /*writeable=*/true);
}

}  // namespace interop

namespace interop {

// Memory-mapped array backed by a file (numpy.memmap). The mapping is created
// with MAP_SHARED so writes in "r+"/"w+" modes are flushed back to the file.
// The Buffer adopts the mapped pointer with owns=false; a custom shared_ptr
// deleter munmaps the region once the last view referencing it is released.
ndarray memmap(const std::string& path, DType dtype, const Shape& shape,
               const std::string& mode) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  int64_t nbytes = n * dtype.itemsize();

  int oflags = 0;
  int prot = 0;
  bool writeable = false;
  if (mode == "r") {
    oflags = O_RDONLY;
    prot = PROT_READ;
    writeable = false;
  } else if (mode == "r+") {
    oflags = O_RDWR;
    prot = PROT_READ | PROT_WRITE;
    writeable = true;
  } else if (mode == "w+") {
    oflags = O_RDWR | O_CREAT | O_TRUNC;
    prot = PROT_READ | PROT_WRITE;
    writeable = true;
  } else {
    throw value_error("memmap: bad mode");
  }

  int fd = open(path.c_str(), oflags, 0644);
  if (fd < 0) throw value_error("memmap: open failed");

  if (mode == "w+") {
    if (ftruncate(fd, nbytes) != 0) {
      close(fd);
      throw value_error("memmap: ftruncate failed");
    }
  }

  void* ptr = mmap(nullptr, static_cast<size_t>(nbytes), prot, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) throw value_error("memmap: mmap failed");

  std::shared_ptr<Buffer> buf(
      new Buffer(static_cast<char*>(ptr), nbytes, false),
      [ptr, nbytes](Buffer* b) {
        delete b;
        munmap(ptr, static_cast<size_t>(nbytes));
      });

  return ndarray(buf, dtype, shape,
                 contiguous_strides(shape, dtype.itemsize(), Order::C), 0,
                 writeable);
}

}  // namespace interop

}  // namespace numpp
