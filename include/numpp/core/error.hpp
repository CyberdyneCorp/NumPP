#pragma once

#include <stdexcept>
#include <string>

#include "numpp/export.hpp"

namespace numpp {

// The exception types carry default visibility (NUMPP_API) so their type_info
// is a single shared symbol across the shared library and its consumers. Without
// this, hidden-visibility type_info differs between libnumpp and the caller on
// macOS/libc++ (which matches by address), and typed `catch` blocks miss
// exceptions thrown from inside the library.

// Base of all recoverable NumPP errors. No CPython C-API, no errno, no abort.
class NUMPP_API error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Typed hierarchy mirroring the NumPy error categories callers already know.
class NUMPP_API value_error : public error {
 public:
  using error::error;
};

class NUMPP_API type_error : public error {
 public:
  using error::error;
};

class NUMPP_API index_error : public error {
 public:
  using error::error;
};

class NUMPP_API axis_error : public error {
 public:
  using error::error;
};

class NUMPP_API linalg_error : public error {
 public:
  using error::error;
};

class NUMPP_API not_implemented_error : public error {
 public:
  using error::error;
};

}  // namespace numpp
