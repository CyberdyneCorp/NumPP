#pragma once
// Minimal self-contained test harness — zero external dependencies so the test
// build works offline and on mobile toolchains. (Catch2/doctest can replace this
// later; kept tiny on purpose for the bootstrap.)

#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace npt {

struct Case { std::string name; std::function<void()> fn; };

inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
inline int& checks() { static int c = 0; return c; }

struct Registrar {
  Registrar(std::string name, std::function<void()> fn) {
    registry().push_back({std::move(name), std::move(fn)});
  }
};

inline void report_fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::fprintf(stderr, "  FAIL %s:%d  %s\n", file, line, msg.c_str());
}

inline int run_all() {
  int passed = 0;
  for (auto& c : registry()) {
    int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      report_fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    }
    if (failures() == before) { ++passed; }
    else { std::fprintf(stderr, "[FAILED] %s\n", c.name.c_str()); }
  }
  std::fprintf(stderr, "\n%d/%zu cases passed, %d checks, %d failures\n",
               passed, registry().size(), checks(), failures());
  return failures() == 0 ? 0 : 1;
}

}  // namespace npt

#define NPT_CONCAT_(a, b) a##b
#define NPT_CONCAT(a, b) NPT_CONCAT_(a, b)
#define TEST_CASE(name)                                                       \
  static void NPT_CONCAT(npt_fn_, __LINE__)();                                \
  static ::npt::Registrar NPT_CONCAT(npt_reg_, __LINE__){name, NPT_CONCAT(npt_fn_, __LINE__)}; \
  static void NPT_CONCAT(npt_fn_, __LINE__)()

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++::npt::checks();                                                       \
    if (!(cond)) ::npt::report_fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_THROWS_AS(expr, Exc)                                           \
  do {                                                                       \
    ++::npt::checks();                                                       \
    bool caught = false;                                                     \
    try { (void)(expr); } catch (const Exc&) { caught = true; } catch (...) {} \
    if (!caught) ::npt::report_fail(__FILE__, __LINE__, "expected " #Exc " from " #expr); \
  } while (0)

#define NPT_MAIN() int main() { return ::npt::run_all(); }
