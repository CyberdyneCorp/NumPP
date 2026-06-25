# NumPP developer tasks — run `just` (or `just --list`) to see everything.
# Requires: cmake >= 3.25, clang++ or g++, ninja (for the gcc/asan recipes),
# python3 + numpy (for the live-oracle tests and example parity checks).

build_dir := "build"
cxx       := "clang++"

# Show the available recipes (default).
default:
    @just --list

# Configure the build. Pass extra cmake flags, e.g.
#   just configure -DNUMPP_WITH_BLAS=ON -DNUMPP_WITH_REFGPU=ON
configure *FLAGS:
    cmake -S . -B {{build_dir}} -DCMAKE_CXX_COMPILER={{cxx}} -DNUMPP_BUILD_EXAMPLES=ON {{FLAGS}}

# Compile the library, test suite, and examples.
build: configure
    cmake --build {{build_dir}}

# Compile only the library.
lib: configure
    cmake --build {{build_dir}} --target numpp

# Run the test suite (unit + integration: validated against live NumPy).
test: build
    ./{{build_dir}}/tests/numpp_tests
alias unit := test

# Run the test suite through CTest.
ctest: build
    ctest --test-dir {{build_dir}} --output-on-failure

# Run the test suite offline (frozen golden data; no Python/NumPy needed).
test-offline: build
    NUMPP_ORACLE_FROZEN=1 ./{{build_dir}}/tests/numpp_tests

# Build and run every example, each self-verifying against NumPy (end-to-end).
examples: build
    #!/usr/bin/env bash
    set -uo pipefail
    fail=0
    for ex in {{build_dir}}/examples/numpp_ee* {{build_dir}}/examples/numpp_nn*; do
      name=$(basename "$ex")
      if out=$("$ex" 2>&1); then echo "✅ $name — $(echo "$out" | tail -1 | xargs)";
      else echo "❌ $name"; echo "$out" | grep -E 'FAIL|ERR'; fail=1; fi
    done
    [ $fail -eq 0 ] && echo "all examples pass NumPy parity" || true
    exit $fail
alias integration := examples

# Run a single example by stem, e.g. `just example nn01_mlp_xor`.
example NAME: build
    ./{{build_dir}}/examples/numpp_{{NAME}}

# Build + test with GCC (separate build dir).
gcc:
    cmake -S . -B build-gcc -G Ninja -DCMAKE_CXX_COMPILER=g++ -DNUMPP_BUILD_EXAMPLES=ON
    cmake --build build-gcc
    ./build-gcc/tests/numpp_tests

# Build + test under AddressSanitizer / UBSan.
asan:
    cmake -S . -B build-asan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DNUMPP_ENABLE_ASAN=ON -DNUMPP_ENABLE_UBSAN=ON
    cmake --build build-asan
    ./build-asan/tests/numpp_tests

# Build with the optional CPU-reference GPU device and run the test suite.
refgpu:
    cmake -S . -B build-refgpu -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DNUMPP_WITH_REFGPU=ON
    cmake --build build-refgpu
    ./build-refgpu/tests/numpp_tests

# Validate all OpenSpec specs.
spec:
    openspec validate --all --strict

# Full local CI: clang tests + gcc + asan + spec + examples.
ci: test gcc asan spec examples

# Install the library + headers to a prefix (default ./_install).
install prefix="_install": build
    cmake --install {{build_dir}} --prefix {{prefix}}

# Remove all build directories.
clean:
    rm -rf {{build_dir}} build-gcc build-asan build-refgpu
