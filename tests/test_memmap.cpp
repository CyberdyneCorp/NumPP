#include "numpp/numpp.hpp"
#include "numpp/interop/interop.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <filesystem>
#include <string>

using namespace numpp;
using namespace numpp::interop;

namespace {
std::string mm_tmp(const std::string& name) {
  return (std::filesystem::temp_directory_path() / name).string();
}
}  // namespace

TEST_CASE("memmap write/read round-trip through a file") {
  using namespace numpp;
  std::string path = mm_tmp("numpp_memmap_rt.bin");
  std::filesystem::remove(path);

  // Write via a w+ mapping in a nested scope so its destructor (munmap) flushes
  // the data to the file before we re-open it.
  {
    ndarray m = interop::memmap(path, kFloat64, {3, 4}, "w+");
    CHECK(m.shape() == Shape({3, 4}));
    for (int64_t i = 0; i < 3; ++i)
      for (int64_t j = 0; j < 4; ++j)
        m.set_item<double>({i, j}, static_cast<double>(i * 4 + j));
  }

  // Read-only mapping should see the written values.
  ndarray r = interop::memmap(path, kFloat64, {3, 4}, "r");
  ndarray expected = arange(0.0, 12.0, 1.0).reshape({3, 4});
  CHECK(allclose(r, expected, 0, 0, true));
  CHECK(r.dtype() == kFloat64);
  CHECK(r.shape() == Shape({3, 4}));

  std::filesystem::remove(path);
}

TEST_CASE("memmap r+ persists in-place modifications") {
  using namespace numpp;
  std::string path = mm_tmp("numpp_memmap_rplus.bin");
  std::filesystem::remove(path);

  { ndarray m = interop::memmap(path, kInt32, {5}, "w+");
    for (int64_t i = 0; i < 5; ++i) m.set_item<int32_t>({i}, static_cast<int32_t>(i)); }

  // Re-open read/write and mutate; changes go to the file.
  { ndarray m = interop::memmap(path, kInt32, {5}, "r+");
    for (int64_t i = 0; i < 5; ++i)
      m.set_item<int32_t>({i}, m.item<int32_t>({i}) * 10); }

  ndarray r = interop::memmap(path, kInt32, {5}, "r");
  ndarray expected = (arange(0.0, 5.0, 1.0) * 10.0).astype(kInt32);
  CHECK(allclose(r, expected, 0, 0, true));
  CHECK(r.item<int32_t>({0}) == 0);
  CHECK(r.item<int32_t>({4}) == 40);

  std::filesystem::remove(path);
}

TEST_CASE("memmap bad mode throws") {
  using namespace numpp;
  std::string path = mm_tmp("numpp_memmap_badmode.bin");
  bool threw = false;
  try {
    interop::memmap(path, kFloat64, {2, 2}, "rb");
  } catch (...) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("memmap reads a numpy-written file (oracle interop)") {
  using namespace numpp;
  if (!npt::numpy_available()) return;
  std::string path = mm_tmp("numpp_memmap_numpy.bin");
  std::filesystem::remove(path);

  // NumPy writes a known array to PATH via tofile (raw, contiguous, C order).
  auto o = npt::oracle(
      "x=np.arange(12.).reshape(3,4); x.tofile(r'" + path + "'); a=x");
  if (!o) return;

  ndarray r = interop::memmap(path, kFloat64, {3, 4}, "r");
  CHECK(allclose(r, *o, 0, 0, true));
  CHECK(r.shape() == o->shape());

  std::filesystem::remove(path);
}
