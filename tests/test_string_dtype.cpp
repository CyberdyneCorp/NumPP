#include "numpp/numpp.hpp"
#include "numpp/strings/string_dtype.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::strings;

TEST_CASE("StringDType round-trips variable-length strings without truncation") {
  StringDType a = StringDType::from_list({"a", "bb", "ccc"});
  CHECK(a.size() == 3);
  CHECK(a.item(0) == "a");
  CHECK(a.item(1) == "bb");
  CHECK(a.item(2) == "ccc");           // longest element kept in full
  CHECK(a.item(2).size() == 3);        // no fixed-width truncation
  CHECK((a.shape() == Shape{3}));
}

TEST_CASE("StringDType 2-D indexing") {
  StringDType a({"a", "bb", "ccc", "dddd"}, Shape{2, 2});
  CHECK(a.item({0, 0}) == "a");
  CHECK(a.item({1, 0}) == "ccc");
  CHECK(a.item({1, 1}) == "dddd");
}

TEST_CASE("StringDType equality matches numpy StringDType ==") {
  StringDType a = StringDType::from_list({"a", "bb", "ccc"});
  StringDType b = StringDType::from_list({"a", "x", "ccc"});
  ndarray got = a.equal(b);  // [true, false, true]
  auto o = npt::oracle(
      "a=(np.array(['a','bb','ccc'],dtype='T')==np.array(['a','x','ccc'],dtype='T'))");
  if (o) CHECK(allclose(got, *o, 0, 0, true));
}

TEST_CASE("StringDType not_equal matches numpy") {
  StringDType a = StringDType::from_list({"a", "bb", "ccc"});
  StringDType b = StringDType::from_list({"a", "x", "ccc"});
  ndarray got = a.not_equal(b);  // [false, true, false]
  auto o = npt::oracle(
      "a=(np.array(['a','bb','ccc'],dtype='T')!=np.array(['a','x','ccc'],dtype='T'))");
  if (o) CHECK(allclose(got, *o, 0, 0, true));
}

TEST_CASE("StringDType concatenation matches numpy.strings.add") {
  // numpy: np.strings.add(['a','bb','ccc'], ['a','x','ccc']) -> ['aa','bbx','cccccc']
  StringDType a = StringDType::from_list({"a", "bb", "ccc"});
  StringDType b = StringDType::from_list({"a", "x", "ccc"});
  StringDType c = a.add(b);
  CHECK(c.item(0) == "aa");
  CHECK(c.item(1) == "bbx");
  CHECK(c.item(2) == "cccccc");
  // Cross-check against numpy via per-element length (lengths are loadable).
  auto o = npt::oracle(
      "r=np.strings.add(np.array(['a','bb','ccc'],dtype='T'),"
      "np.array(['a','x','ccc'],dtype='T')); a=np.strings.str_len(r).astype('int64')");
  if (o) {
    CHECK((static_cast<int64_t>(c.item(0).size()) == o->item<int64_t>({0})));
    CHECK((static_cast<int64_t>(c.item(1).size()) == o->item<int64_t>({1})));
    CHECK((static_cast<int64_t>(c.item(2).size()) == o->item<int64_t>({2})));
  }
}

TEST_CASE("StringDType broadcasts a single-element operand") {
  StringDType a = StringDType::from_list({"x", "y", "z"});
  StringDType suf = StringDType::from_list({"_"});
  StringDType c = a.add(suf);
  CHECK(c.item(0) == "x_");
  CHECK(c.item(2) == "z_");
}

TEST_CASE("StringDType set_item replaces an element") {
  StringDType a = StringDType::from_list({"a", "bb", "ccc"});
  a.set_item(1, "replaced");
  CHECK(a.item(1) == "replaced");
}
