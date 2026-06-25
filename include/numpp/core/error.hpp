#pragma once

#include <stdexcept>
#include <string>

namespace numpp {

// Base of all recoverable NumPP errors. No CPython C-API, no errno, no abort.
class error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Typed hierarchy mirroring the NumPy error categories callers already know.
class value_error : public error {
 public:
  using error::error;
};

class type_error : public error {
 public:
  using error::error;
};

class index_error : public error {
 public:
  using error::error;
};

class axis_error : public error {
 public:
  using error::error;
};

class linalg_error : public error {
 public:
  using error::error;
};

class not_implemented_error : public error {
 public:
  using error::error;
};

}  // namespace numpp
