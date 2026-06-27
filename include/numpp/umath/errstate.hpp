#pragma once

#include <functional>
#include <map>
#include <string>

#include "numpp/core/error.hpp"
#include "numpp/export.hpp"

// numpy.errstate / seterr / geterr: control how floating-point conditions
// (divide, over, under, invalid) are handled by element-wise float kernels.
// Policies: "ignore", "warn", "raise", "call", "print", "log". The thread-local
// state is consulted by the binary/unary float ufuncs after each operation.

namespace numpp {

// Raised when an FP condition occurs and its policy is "raise" (numpy uses
// FloatingPointError, a subclass of ArithmeticError).
class NUMPP_API floating_point_error : public error {
 public:
  explicit floating_point_error(const std::string& what) : error(what) {}
};

// Read the current per-condition policies (keys: "divide", "over", "under",
// "invalid"), matching numpy.geterr().
NUMPP_API std::map<std::string, std::string> geterr();

// Set policies and return the previous full state. Accepted keys: "all"
// (applies to every condition), "divide", "over", "under", "invalid". Values
// must be one of the valid policy names. Matches numpy.seterr(**kwargs).
NUMPP_API std::map<std::string, std::string> seterr(
    const std::map<std::string, std::string>& kwargs);

// Convenience: set every condition to the same policy.
NUMPP_API std::map<std::string, std::string> seterr(const std::string& all);

// Install the callback used by the "call" policy; invoked as cb(condition,
// flag) where condition is "divide"/"over"/"under"/"invalid". Matches
// numpy.seterrcall (callable form).
NUMPP_API void seterrcall(std::function<void(const std::string&, int)> cb);

// Scoped guard: applies kwargs on construction, restores the prior state on
// destruction. Mirrors the numpy.errstate context manager.
class NUMPP_API errstate {
 public:
  explicit errstate(const std::map<std::string, std::string>& kwargs);
  explicit errstate(const std::string& all);
  ~errstate();
  errstate(const errstate&) = delete;
  errstate& operator=(const errstate&) = delete;

 private:
  std::map<std::string, std::string> prev_;
};

namespace detail {
// FP condition flags (bit set), independent of <cfenv> macro values.
enum FpFlag { kFpDivide = 1, kFpOver = 2, kFpUnder = 4, kFpInvalid = 8 };

// Helper for kernels: clears the host FP exception flags on construction.
// Call check() after the compute loop to test the flags and apply the active
// errstate policy (may throw floating_point_error). The destructor only clears
// flags and never throws (destructors are noexcept).
class NUMPP_API fp_guard {
 public:
  fp_guard();
  ~fp_guard();
  void check();
  fp_guard(const fp_guard&) = delete;
  fp_guard& operator=(const fp_guard&) = delete;
};
}  // namespace detail

}  // namespace numpp
