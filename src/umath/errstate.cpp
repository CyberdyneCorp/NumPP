#include "numpp/umath/errstate.hpp"

#include <array>
#include <cfenv>
#include <cstdio>
#include <stdexcept>

namespace numpp {
namespace {

// The four conditions, in numpy's dict order.
constexpr std::array<const char*, 4> kConditions{"divide", "over", "under", "invalid"};

bool valid_policy(const std::string& p) {
  return p == "ignore" || p == "warn" || p == "raise" || p == "call" ||
         p == "print" || p == "log";
}

// Thread-local policy per condition (default "warn", matching numpy's default
// for divide/over/invalid; numpy defaults under to "ignore").
struct State {
  std::array<std::string, 4> policy{"warn", "warn", "ignore", "warn"};
  std::function<void(const std::string&, int)> callback;
};

State& state() {
  thread_local State s;
  return s;
}

std::map<std::string, std::string> snapshot() {
  std::map<std::string, std::string> m;
  for (std::size_t i = 0; i < kConditions.size(); ++i) m[kConditions[i]] = state().policy[i];
  return m;
}

void apply_kwargs(const std::map<std::string, std::string>& kw) {
  // "all" first, then specific overrides — matches numpy.seterr.
  auto it = kw.find("all");
  if (it != kw.end()) {
    if (!valid_policy(it->second)) throw value_error("invalid error-state policy: " + it->second);
    for (auto& p : state().policy) p = it->second;
  }
  for (std::size_t i = 0; i < kConditions.size(); ++i) {
    auto f = kw.find(kConditions[i]);
    if (f == kw.end()) continue;
    if (!valid_policy(f->second)) throw value_error("invalid error-state policy: " + f->second);
    state().policy[i] = f->second;
  }
  for (const auto& [k, v] : kw) {
    (void)v;
    if (k != "all" && k != "divide" && k != "over" && k != "under" && k != "invalid")
      throw value_error("unknown error-state key: " + k);
  }
}

}  // namespace

std::map<std::string, std::string> geterr() { return snapshot(); }

std::map<std::string, std::string> seterr(const std::map<std::string, std::string>& kwargs) {
  auto prev = snapshot();
  apply_kwargs(kwargs);
  return prev;
}

std::map<std::string, std::string> seterr(const std::string& all) {
  return seterr(std::map<std::string, std::string>{{"all", all}});
}

void seterrcall(std::function<void(const std::string&, int)> cb) {
  state().callback = std::move(cb);
}

errstate::errstate(const std::map<std::string, std::string>& kwargs) : prev_(snapshot()) {
  apply_kwargs(kwargs);
}
errstate::errstate(const std::string& all)
    : errstate(std::map<std::string, std::string>{{"all", all}}) {}
errstate::~errstate() {
  for (std::size_t i = 0; i < kConditions.size(); ++i) state().policy[i] = prev_[kConditions[i]];
}

namespace detail {
namespace {

// Map host <cfenv> exception bits to our condition index + flag.
struct Cond { int fe; int idx; const char* name; int flag; };
const std::array<Cond, 4> kMap{{
    {FE_DIVBYZERO, 0, "divide", kFpDivide},
    {FE_OVERFLOW, 1, "over", kFpOver},
    {FE_UNDERFLOW, 2, "under", kFpUnder},
    {FE_INVALID, 3, "invalid", kFpInvalid},
}};

void handle(int raised) {
  if (raised == 0) return;
  for (const auto& c : kMap) {
    if ((raised & c.fe) == 0) continue;
    const std::string& pol = state().policy[c.idx];
    if (pol == "ignore") continue;
    if (pol == "raise")
      throw floating_point_error(std::string(c.name) + " encountered in floating-point operation");
    if (pol == "call") {
      if (state().callback) state().callback(c.name, c.flag);
      continue;
    }
    // "warn", "print" and "log" all emit a diagnostic; we route to stderr.
    std::fprintf(stderr, "RuntimeWarning: %s encountered\n", c.name);
  }
}

}  // namespace

fp_guard::fp_guard() { std::feclearexcept(FE_ALL_EXCEPT); }
fp_guard::~fp_guard() { std::feclearexcept(FE_ALL_EXCEPT); }
void fp_guard::check() {
  int raised = std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
  std::feclearexcept(FE_ALL_EXCEPT);
  handle(raised);  // may throw floating_point_error when policy is "raise"
}

}  // namespace detail
}  // namespace numpp
