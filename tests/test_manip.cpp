#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// ---- joining ----
TEST_CASE("concatenate axis0 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 12., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.concatenate([np.arange(6.).reshape(2,3), np.arange(6,12.).reshape(2,3)],axis=0)");
  if (o) CHECK(allclose(concatenate({x, y}, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("concatenate axis1 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 12., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.concatenate([np.arange(6.).reshape(2,3), np.arange(6,12.).reshape(2,3)],axis=1)");
  if (o) CHECK(allclose(concatenate({x, y}, 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("concatenate negative axis vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(100., 102., 1., kFloat64).reshape({2, 1});
  auto o = npt::oracle("a=np.concatenate([np.arange(6.).reshape(2,3), np.arange(100,102.).reshape(2,1)],axis=-1)");
  if (o) CHECK(allclose(concatenate({x, y}, -1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("concatenate dtype promotion int+float") {
  ndarray x = arange(0, 3, 1, kInt64);
  ndarray y = arange(3., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.concatenate([np.arange(3), np.arange(3,6.)])");
  if (o) {
    ndarray r = concatenate({x, y}, 0);
    CHECK(r.dtype() == kFloat64);
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("stack axis0 vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  ndarray y = arange(3., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.stack([np.arange(3.), np.arange(3,6.)],axis=0)");
  if (o) CHECK(allclose(stack({x, y}, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("stack axis-1 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 12., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.stack([np.arange(6.).reshape(2,3), np.arange(6,12.).reshape(2,3)],axis=-1)");
  if (o) CHECK(allclose(stack({x, y}, -1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("hstack 1d vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  ndarray y = arange(3., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.hstack([np.arange(3.), np.arange(3,6.)])");
  if (o) CHECK(allclose(hstack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("hstack 2d vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 10., 1., kFloat64).reshape({2, 2});
  auto o = npt::oracle("a=np.hstack([np.arange(6.).reshape(2,3), np.arange(6,10.).reshape(2,2)])");
  if (o) CHECK(allclose(hstack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("vstack 1d promotes to 2d vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  ndarray y = arange(3., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.vstack([np.arange(3.), np.arange(3,6.)])");
  if (o) CHECK(allclose(vstack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("vstack 2d vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 9., 1., kFloat64).reshape({1, 3});
  auto o = npt::oracle("a=np.vstack([np.arange(6.).reshape(2,3), np.arange(6,9.).reshape(1,3)])");
  if (o) CHECK(allclose(vstack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("dstack vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 12., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.dstack([np.arange(6.).reshape(2,3), np.arange(6,12.).reshape(2,3)])");
  if (o) CHECK(allclose(dstack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("column_stack 1d vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  ndarray y = arange(3., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.column_stack((np.arange(3.), np.arange(3,6.)))");
  if (o) CHECK(allclose(column_stack({x, y}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("column_stack 2d vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({3, 2});
  ndarray y = arange(6., 9., 1., kFloat64).reshape({3, 1});
  auto o = npt::oracle("a=np.column_stack((np.arange(6.).reshape(3,2), np.arange(6,9.).reshape(3,1)))");
  if (o) CHECK(allclose(column_stack({x, y}), *o, 1e-9, 1e-12, true));
}

// ---- splitting ----
TEST_CASE("array_split uneven 1d vs numpy") {
  ndarray x = arange(0., 7., 1., kFloat64);
  auto parts = array_split(x, 3);  // sizes 3,2,2
  CHECK(parts.size() == 3u);
  auto o0 = npt::oracle("a=np.array_split(np.arange(7.),3)[0]");
  auto o1 = npt::oracle("a=np.array_split(np.arange(7.),3)[1]");
  auto o2 = npt::oracle("a=np.array_split(np.arange(7.),3)[2]");
  if (o0) CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
  if (o1) CHECK(allclose(parts[1], *o1, 1e-9, 1e-12, true));
  if (o2) CHECK(allclose(parts[2], *o2, 1e-9, 1e-12, true));
}

TEST_CASE("array_split more sections than elements") {
  ndarray x = arange(0., 2., 1., kFloat64);
  auto parts = array_split(x, 4);  // sizes 1,1,0,0
  CHECK(parts.size() == 4u);
  CHECK(parts[2].size() == 0);
  CHECK(parts[3].size() == 0);
  auto o = npt::oracle("a=np.array_split(np.arange(2.),4)[1]");
  if (o) CHECK(allclose(parts[1], *o, 1e-9, 1e-12, true));
}

TEST_CASE("array_split int dtype preserved") {
  ndarray x = arange(0, 10, 1, kInt64);
  auto parts = array_split(x, 4);  // sizes 3,3,2,2
  CHECK(parts[0].dtype() == kInt64);
  auto o0 = npt::oracle("a=np.array_split(np.arange(10),4)[0]");
  auto o3 = npt::oracle("a=np.array_split(np.arange(10),4)[3]");
  if (o0) CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
  if (o3) CHECK(allclose(parts[3], *o3, 1e-9, 1e-12, true));
}

TEST_CASE("split even axis0 vs numpy") {
  ndarray x = arange(0., 8., 1., kFloat64).reshape({4, 2});
  auto parts = split(x, 2, 0);
  CHECK(parts.size() == 2u);
  auto o0 = npt::oracle("a=np.split(np.arange(8.).reshape(4,2),2,axis=0)[0]");
  auto o1 = npt::oracle("a=np.split(np.arange(8.).reshape(4,2),2,axis=0)[1]");
  if (o0) CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
  if (o1) CHECK(allclose(parts[1], *o1, 1e-9, 1e-12, true));
}

TEST_CASE("split negative axis vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({2, 6});
  auto parts = split(x, 3, -1);  // along axis 1 -> three 2x2 blocks
  CHECK(parts.size() == 3u);
  auto o = npt::oracle("a=np.split(np.arange(12.).reshape(2,6),3,axis=-1)[1]");
  if (o) CHECK(allclose(parts[1], *o, 1e-9, 1e-12, true));
}

TEST_CASE("split uneven division throws") {
  ndarray x = arange(0., 7., 1., kFloat64);
  CHECK_THROWS_AS(split(x, 3), value_error);
}

TEST_CASE("hsplit 2d vs numpy") {
  ndarray x = arange(0., 16., 1., kFloat64).reshape({4, 4});
  auto parts = hsplit(x, 2);
  CHECK(parts.size() == 2u);
  auto o0 = npt::oracle("a=np.hsplit(np.arange(16.).reshape(4,4),2)[0]");
  auto o1 = npt::oracle("a=np.hsplit(np.arange(16.).reshape(4,4),2)[1]");
  if (o0) CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
  if (o1) CHECK(allclose(parts[1], *o1, 1e-9, 1e-12, true));
}

TEST_CASE("hsplit 1d uses axis 0") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto parts = hsplit(x, 3);
  CHECK(parts.size() == 3u);
  auto o = npt::oracle("a=np.hsplit(np.arange(6.),3)[2]");
  if (o) CHECK(allclose(parts[2], *o, 1e-9, 1e-12, true));
}

TEST_CASE("vsplit 2d vs numpy") {
  ndarray x = arange(0., 16., 1., kFloat64).reshape({4, 4});
  auto parts = vsplit(x, 2);
  CHECK(parts.size() == 2u);
  auto o0 = npt::oracle("a=np.vsplit(np.arange(16.).reshape(4,4),2)[0]");
  auto o1 = npt::oracle("a=np.vsplit(np.arange(16.).reshape(4,4),2)[1]");
  if (o0) CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
  if (o1) CHECK(allclose(parts[1], *o1, 1e-9, 1e-12, true));
}

TEST_CASE("vsplit on 1d throws") {
  ndarray x = arange(0., 6., 1., kFloat64);
  CHECK_THROWS_AS(vsplit(x, 3), value_error);
}

// ---- tiling ----
TEST_CASE("tile 1-D scalar reps vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.tile(np.arange(3.), 2)");
  if (o) CHECK(allclose(tile(x, {2}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("tile 1-D with 2-D reps vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.tile(np.arange(3.), (2,2))");
  if (o) CHECK(allclose(tile(x, {2, 2}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("tile 2-D with 2-D reps vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.tile(np.arange(6.).reshape(2,3), (2,3))");
  if (o) CHECK(allclose(tile(x, {2, 3}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("tile 2-D with shorter reps (prepended dims) vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.tile(np.arange(6.).reshape(2,3), 2)");
  if (o) CHECK(allclose(tile(x, {2}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("tile reps longer than ndim vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.tile(np.arange(3.), (2,1,3))");
  if (o) CHECK(allclose(tile(x, {2, 1, 3}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("tile int dtype vs numpy") {
  ndarray x = arange(0., 4., 1., kInt64).reshape({2, 2});
  auto o = npt::oracle("a=np.tile(np.arange(4).reshape(2,2), (2,2))");
  if (o) CHECK(allclose(tile(x, {2, 2}), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat axis=None flatten vs numpy") {
  ndarray x = arange(0., 3., 1., kFloat64);
  auto o = npt::oracle("a=np.repeat(np.arange(3.), 2)");
  if (o) CHECK(allclose(repeat(x, 2), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat 2-D axis=0 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.repeat(np.arange(6.).reshape(2,3), 2, axis=0)");
  if (o) CHECK(allclose(repeat(x, 2, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat 2-D axis=1 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.repeat(np.arange(6.).reshape(2,3), 3, axis=1)");
  if (o) CHECK(allclose(repeat(x, 3, 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat negative axis vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.repeat(np.arange(6.).reshape(2,3), 2, axis=-1)");
  if (o) CHECK(allclose(repeat(x, 2, -1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat 2-D axis=None vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.repeat(np.arange(6.).reshape(2,3), 2)");
  if (o) CHECK(allclose(repeat(x, 2), *o, 1e-9, 1e-12, true));
}

TEST_CASE("repeat int dtype axis=0 vs numpy") {
  ndarray x = arange(0., 4., 1., kInt64).reshape({2, 2});
  auto o = npt::oracle("a=np.repeat(np.arange(4).reshape(2,2), 3, axis=0)");
  if (o) CHECK(allclose(repeat(x, 3, 0), *o, 1e-9, 1e-12, true));
}

// ---- rearranging ----
TEST_CASE("flip axis0 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.flip(np.arange(12.).reshape(3,4),0)");
  if (o) CHECK(allclose(flip(x, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("flip axis1 negative vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.flip(np.arange(12.).reshape(3,4),-1)");
  if (o) CHECK(allclose(flip(x, -1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("flip all axes vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o = npt::oracle("a=np.flip(np.arange(24.).reshape(2,3,4))");
  if (o) CHECK(allclose(flip(x), *o, 1e-9, 1e-12, true));
}

TEST_CASE("fliplr and flipud vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o1 = npt::oracle("a=np.fliplr(np.arange(12.).reshape(3,4))");
  if (o1) CHECK(allclose(fliplr(x), *o1, 1e-9, 1e-12, true));
  auto o2 = npt::oracle("a=np.flipud(np.arange(12.).reshape(3,4))");
  if (o2) CHECK(allclose(flipud(x), *o2, 1e-9, 1e-12, true));
}

TEST_CASE("roll along axis1 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.roll(np.arange(12.).reshape(3,4),2,axis=1)");
  if (o) CHECK(allclose(roll(x, 2, 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("roll negative shift axis0 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.roll(np.arange(12.).reshape(3,4),-1,axis=0)");
  if (o) CHECK(allclose(roll(x, -1, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("roll flatten (axis None) vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.roll(np.arange(12.).reshape(3,4),5)");
  if (o) CHECK(allclose(roll(x, 5), *o, 1e-9, 1e-12, true));
}

TEST_CASE("rot90 k=1 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.rot90(np.arange(12.).reshape(3,4),1)");
  if (o) CHECK(allclose(rot90(x, 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("rot90 k=2 and negative k vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o2 = npt::oracle("a=np.rot90(np.arange(12.).reshape(3,4),2)");
  if (o2) CHECK(allclose(rot90(x, 2), *o2, 1e-9, 1e-12, true));
  auto o3 = npt::oracle("a=np.rot90(np.arange(12.).reshape(3,4),-1)");
  if (o3) CHECK(allclose(rot90(x, -1), *o3, 1e-9, 1e-12, true));
}

TEST_CASE("moveaxis 0 to -1 vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o = npt::oracle("a=np.moveaxis(np.arange(24.).reshape(2,3,4),0,-1)");
  if (o) CHECK(allclose(moveaxis(x, 0, -1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("moveaxis 2 to 0 int dtype vs numpy") {
  ndarray x = arange(0., 24., 1.).reshape({2, 3, 4});  // int64
  auto o = npt::oracle("a=np.moveaxis(np.arange(24).reshape(2,3,4),2,0)");
  if (o) CHECK(allclose(moveaxis(x, 2, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("atleast_1d/2d/3d on 1d vs numpy") {
  ndarray v = arange(0., 3., 1., kFloat64);  // shape (3,)
  auto o1 = npt::oracle("a=np.atleast_1d(np.arange(3.))");
  if (o1) CHECK(allclose(atleast_1d(v), *o1, 1e-9, 1e-12, true));
  auto o2 = npt::oracle("a=np.atleast_2d(np.arange(3.))");
  if (o2) CHECK(allclose(atleast_2d(v), *o2, 1e-9, 1e-12, true));
  auto o3 = npt::oracle("a=np.atleast_3d(np.arange(3.))");
  if (o3) CHECK(allclose(atleast_3d(v), *o3, 1e-9, 1e-12, true));
}

TEST_CASE("atleast_3d from 2d vs numpy") {
  ndarray m = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.atleast_3d(np.arange(6.).reshape(2,3))");
  if (o) CHECK(allclose(atleast_3d(m), *o, 1e-9, 1e-12, true));
}

// ---- editing ----
// Helper: build a 1-D float array from a literal list (then cast).
static ndarray ed_vec(const std::vector<double>& v, DType dt = kFloat64) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64);
  for (size_t i = 0; i < v.size(); ++i) a.set_item<double>({static_cast<int64_t>(i)}, v[i]);
  return a.astype(dt);
}

TEST_CASE("append along axis 0 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray y = arange(6., 9., 1., kFloat64).reshape({1, 3});
  auto o = npt::oracle("a=np.append(np.arange(6.).reshape(2,3),np.arange(6.,9.).reshape(1,3),axis=0)");
  if (o) CHECK(allclose(append(x, y, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("append axis=None flattens both operands") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.append(np.arange(6.).reshape(2,3),np.array([7.,8.]))");
  if (o) CHECK(allclose(append(x, ed_vec({7., 8.}), std::nullopt), *o, 1e-9, 1e-12, true));
}

TEST_CASE("append promotes dtype like numpy") {
  ndarray x = arange(0., 3., 1., kInt64);   // int64
  ndarray y = ed_vec({3.5, 4.5});           // float64
  auto o = npt::oracle("a=np.append(np.arange(3),np.array([3.5,4.5]))");
  if (o) CHECK(allclose(append(x, y, std::nullopt), *o, 1e-9, 1e-12, true));
}

TEST_CASE("insert axis=None before index") {
  auto o = npt::oracle("a=np.insert(np.arange(6.),2,99.)");
  if (o) CHECK(allclose(insert(arange(0., 6., 1., kFloat64), 2, ed_vec({99.}), std::nullopt),
                        *o, 1e-9, 1e-12, true));
}

TEST_CASE("insert scalar broadcasts along axis 1") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.insert(np.arange(6.).reshape(2,3),1,99.,axis=1)");
  if (o) CHECK(allclose(insert(x, 1, ed_vec({99.}), 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("insert vector column along axis 1") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.insert(np.arange(6.).reshape(2,3),1,[97.,98.],axis=1)");
  if (o) CHECK(allclose(insert(x, 1, ed_vec({97., 98.}), 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("insert row along axis 0 and negative index") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o0 = npt::oracle("a=np.insert(np.arange(6.).reshape(2,3),0,[10.,11.,12.],axis=0)");
  if (o0) CHECK(allclose(insert(x, 0, ed_vec({10., 11., 12.}), 0), *o0, 1e-9, 1e-12, true));
  auto on = npt::oracle("a=np.insert(np.arange(6.).reshape(2,3),-1,99.,axis=1)");
  if (on) CHECK(allclose(insert(x, -1, ed_vec({99.}), 1), *on, 1e-9, 1e-12, true));
}

TEST_CASE("delete along axes and flattened") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o0 = npt::oracle("a=np.delete(np.arange(12.).reshape(3,4),1,axis=0)");
  if (o0) CHECK(allclose(delete_(x, 1, 0), *o0, 1e-9, 1e-12, true));
  auto o1 = npt::oracle("a=np.delete(np.arange(12.).reshape(3,4),2,axis=1)");
  if (o1) CHECK(allclose(delete_(x, 2, 1), *o1, 1e-9, 1e-12, true));
  auto of = npt::oracle("a=np.delete(np.arange(6.),3)");
  if (of) CHECK(allclose(delete_(arange(0., 6., 1., kFloat64), 3, std::nullopt), *of, 1e-9, 1e-12, true));
}

TEST_CASE("delete with negative index") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.delete(np.arange(6.).reshape(2,3),-1,axis=1)");
  if (o) CHECK(allclose(delete_(x, -1, 1), *o, 1e-9, 1e-12, true));
}

TEST_CASE("resize cycles flattened data (grow and shrink)") {
  auto ob = npt::oracle("a=np.resize(np.arange(6.),(2,5))");
  if (ob) CHECK(allclose(resize(arange(0., 6., 1., kFloat64), {2, 5}), *ob, 1e-9, 1e-12, true));
  auto os = npt::oracle("a=np.resize(np.arange(6.),(2,2))");
  if (os) CHECK(allclose(resize(arange(0., 6., 1., kFloat64), {2, 2}), *os, 1e-9, 1e-12, true));
}

TEST_CASE("pad constant mode places original in the center") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.pad(np.arange(6.).reshape(2,3),((1,2),(2,1)),constant_values=7.)");
  if (o) CHECK(allclose(pad(x, {{1, 2}, {2, 1}}, "constant", 7.0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("pad edge mode replicates borders and corners") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.pad(np.arange(6.).reshape(2,3),((1,2),(2,1)),mode='edge')");
  if (o) CHECK(allclose(pad(x, {{1, 2}, {2, 1}}, "edge", 0.0), *o, 1e-9, 1e-12, true));
}
