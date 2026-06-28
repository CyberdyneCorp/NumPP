#!/usr/bin/env bash
# Downstream-consumption smoke test — the regression guard for the packaging fixes
# in #107/#108/#109. Exercises both supported consumption modes against a from-source
# build:
#   1. install to a prefix, then consume via find_package(NumPP)   (#107)
#   2. embed the source tree via add_subdirectory                  (#108)
# The from-source NumPP build uses -Werror to keep maintainer builds strict (#109.1).
#
# Usage: tests/packaging/smoke.sh [<cxx-compiler>]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${1:-${CXX:-c++}}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PREFIX="$WORK/prefix"

echo "== repo: $REPO"
echo "== cxx:  $CXX"
echo "== work: $WORK"

# --- Build + install NumPP to a prefix (strict: warnings-as-errors on) ----------
echo "== [1/3] configure + build + install NumPP -> $PREFIX"
cmake -S "$REPO" -B "$WORK/build" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNUMPP_BUILD_TESTS=OFF \
  -DNUMPP_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" >/dev/null
cmake --build "$WORK/build" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" >/dev/null
cmake --install "$WORK/build" >/dev/null

# The C public header must have been installed (#107).
test -f "$PREFIX/include/numpp/interop/dlpack.h" \
  || { echo "FAIL: numpp/interop/dlpack.h missing from install prefix (#107)"; exit 1; }
echo "   installed dlpack.h present (#107 ok)"

# --- Mode 1: find_package from the installed prefix (#107) -----------------------
echo "== [2/3] consume via find_package(NumPP)"
cmake -S "$REPO/tests/packaging/consumer" -B "$WORK/c_fp" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$WORK/c_fp" >/dev/null
LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}" "$WORK/c_fp/consumer"

# --- Mode 2: add_subdirectory embedding (#108) ----------------------------------
echo "== [3/3] consume via add_subdirectory"
cmake -S "$REPO/tests/packaging/embed" -B "$WORK/c_embed" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DNUMPP_SOURCE_DIR="$REPO" >/dev/null
cmake --build "$WORK/c_embed" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" >/dev/null
"$WORK/c_embed/embed"

echo "== PASS: NumPP consumable via find_package and add_subdirectory"
