#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

// Build a 3x4 float64 array from arange(12).
static ndarray build_data_3x4() {
  ndarray d(numpp::Shape{3, 4}, numpp::kFloat64);
  double v = 0.0;
  for (int64_t i = 0; i < 3; ++i)
    for (int64_t j = 0; j < 4; ++j) d.set_item<double>({i, j}, v++);
  return d;
}

// Mask M = [[0,1,0,0],[0,0,1,0],[1,1,1,1]] (third row fully masked).
static ndarray build_mask_3x4() {
  int m[3][4] = {{0, 1, 0, 0}, {0, 0, 1, 0}, {1, 1, 1, 1}};
  ndarray mk(numpp::Shape{3, 4}, numpp::kBool);
  for (int64_t i = 0; i < 3; ++i)
    for (int64_t j = 0; j < 4; ++j)
      mk.set_item<bool>({i, j}, m[i][j] != 0);
  return mk;
}

static const char* kSetup =
    "D=np.arange(12.).reshape(3,4);"
    "M=[[0,1,0,0],[0,0,1,0],[1,1,1,1]];"
    "A=np.ma.masked_array(D,mask=M);";

TEST_CASE("ma reduce axis vs numpy") {
  using namespace numpp;
  using namespace numpp::ma;
  MaskedArray A = masked_array(build_data_3x4(), build_mask_3x4());

  // sum axis=0 (data + mask). Column index 3? No column is all-masked here,
  // but along axis=0 no column is fully masked, so all output unmasked.
  {
    MaskedArray R = sum_axis(A, 0);
    auto o = npt::oracle(std::string(kSetup) + "a=A.sum(axis=0).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
    auto om = npt::oracle(std::string(kSetup) +
                          "a=np.ma.getmaskarray(A.sum(axis=0)).astype(np.float64)");
    if (om) CHECK(allclose(R.mask.astype(kFloat64), *om, 0, 0, true));
  }

  // sum axis=1: row 2 fully masked -> masked output element.
  {
    MaskedArray R = sum_axis(A, 1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.sum(axis=1).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
    auto om = npt::oracle(std::string(kSetup) +
                          "a=np.ma.getmaskarray(A.sum(axis=1)).astype(np.float64)");
    if (om) CHECK(allclose(R.mask.astype(kFloat64), *om, 0, 0, true));
  }

  // mean axis=1 (row 2 masked).
  {
    MaskedArray R = mean_axis(A, 1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.mean(axis=1).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
    auto om = npt::oracle(std::string(kSetup) +
                          "a=np.ma.getmaskarray(A.mean(axis=1)).astype(np.float64)");
    if (om) CHECK(allclose(R.mask.astype(kFloat64), *om, 0, 0, true));
  }

  // max axis=1.
  {
    MaskedArray R = max_axis(A, 1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.max(axis=1).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
  }

  // min axis=0.
  {
    MaskedArray R = min_axis(A, 0);
    auto o = npt::oracle(std::string(kSetup) + "a=A.min(axis=0).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
  }

  // prod axis=1.
  {
    MaskedArray R = prod_axis(A, 1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.prod(axis=1).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
  }

  // count axis=1 (plain int64, not masked).
  {
    ndarray c = count_axis(A, 1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.count(axis=1).astype(np.int64)");
    if (o) CHECK(allclose(c.astype(kFloat64), o->astype(kFloat64), 0, 0, true));
  }

  // count axis=0.
  {
    ndarray c = count_axis(A, 0);
    auto o = npt::oracle(std::string(kSetup) + "a=A.count(axis=0).astype(np.int64)");
    if (o) CHECK(allclose(c.astype(kFloat64), o->astype(kFloat64), 0, 0, true));
  }

  // negative axis support: axis=-1 same as axis=1.
  {
    MaskedArray R = sum_axis(A, -1);
    auto o = npt::oracle(std::string(kSetup) + "a=A.sum(axis=-1).filled(-1.)");
    if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("ma reduce fully-masked column produces masked output") {
  using namespace numpp;
  using namespace numpp::ma;
  // Make column 0 fully masked so axis=0 yields a masked output element.
  ndarray d(Shape{2, 3}, kFloat64);
  double v = 1.0;
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) d.set_item<double>({i, j}, v++);
  int mm[2][3] = {{1, 0, 0}, {1, 0, 1}};
  ndarray mk(Shape{2, 3}, kBool);
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) mk.set_item<bool>({i, j}, mm[i][j] != 0);
  MaskedArray A = masked_array(d, mk);

  const char* setup =
      "D=np.array([[1.,2.,3.],[4.,5.,6.]]);"
      "M=[[1,0,0],[1,0,1]];"
      "A=np.ma.masked_array(D,mask=M);";

  MaskedArray R = sum_axis(A, 0);
  auto o = npt::oracle(std::string(setup) + "a=A.sum(axis=0).filled(-1.)");
  if (o) CHECK(allclose(filled(R, -1.0), *o, 1e-9, 1e-12, true));
  auto om = npt::oracle(std::string(setup) +
                        "a=np.ma.getmaskarray(A.sum(axis=0)).astype(np.float64)");
  if (om) CHECK(allclose(R.mask.astype(kFloat64), *om, 0, 0, true));
}
