#pragma once

#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/interop/dlpack.h"

namespace numpp {
namespace interop {

// DLPack export/import. to_dlpack returns a heap DLManagedTensor that keeps the
// source array's buffer alive until its deleter is invoked. from_dlpack adopts a
// DLManagedTensor (zero-copy) and calls its deleter when the resulting array's
// buffer is released.
NUMPP_API DLManagedTensor* to_dlpack(const ndarray& a);
NUMPP_API ndarray from_dlpack(DLManagedTensor* tensor);

// Memory-mapped array backed by a file (numpy.memmap). mode: "r" (read-only),
// "r+" (read/write existing), "w+" (create/truncate to shape). Writes in r+/w+
// modes go to the file. The mapping is unmapped when the last view is released.
NUMPP_API ndarray memmap(const std::string& path, DType dtype, const Shape& shape,
                         const std::string& mode = "r+");

}  // namespace interop
}  // namespace numpp
