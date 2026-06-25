#pragma once

#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// NPY format (v1.0/v2.0). save writes a C-contiguous array; load reads any dtype
// and honors fortran_order. Matches numpy.save / numpy.load on disk.
NUMPP_API void save(const std::string& path, const ndarray& a);
NUMPP_API ndarray load(const std::string& path);

// Serialize/parse an NPY image in memory (used by save/load and by .npz).
NUMPP_API std::string npy_bytes(const ndarray& a);
NUMPP_API ndarray npy_from_bytes(const char* data, size_t size);

// dtype <-> NPY descr string ("<f8", "|b1", "<c16", ...).
NUMPP_API std::string dtype_to_descr(DType d);
NUMPP_API DType descr_to_dtype(const std::string& descr);

}  // namespace numpp
