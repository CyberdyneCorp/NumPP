# build-packaging Specification

## ADDED Requirements

### Requirement: CMake C++20 project

NumPP SHALL build with CMake (>= 3.25) targeting C++20, compilable with clang
(primary) and GCC. The build SHALL expose options mirroring the sibling SymPP
project: shared/static selection, build tests, build examples, build benchmarks,
build docs, AddressSanitizer, UndefinedBehaviorSanitizer, and warnings-as-errors.

#### Scenario: Configure and build core
- WHEN the project is configured with default options and built with clang
- THEN the core library compiles in C++20 with no errors

#### Scenario: Sanitizer build
- WHEN configured with the ASan option enabled
- THEN the test suite builds and runs under AddressSanitizer

### Requirement: Zero required runtime dependencies for core

The NumPP core library SHALL have no mandatory third-party runtime dependency.
BLAS, LAPACK, and all GPU backends SHALL be optional features that are absent
from the dependency closure of a default (core-only) build.

#### Scenario: Core links with no external numeric library
- WHEN the core library is built with default options
- THEN it links without requiring BLAS, LAPACK, or any GPU SDK

### Requirement: Public header layout and umbrella header

NumPP SHALL install public headers under `include/numpp/<module>/` and SHALL
provide an umbrella header `include/numpp/numpp.hpp` that transitively includes
the public API. A consuming project SHALL be able to `find_package(NumPP)` and
link an exported target.

#### Scenario: Single-include usage
- GIVEN a consumer that includes only `numpp/numpp.hpp` and links the exported target
- WHEN it constructs an `ndarray` and performs a basic operation
- THEN it compiles and runs against the installed package

### Requirement: vcpkg and Conan packaging

NumPP SHALL ship a `vcpkg.json` manifest and a `conanfile.py` so the library can
be consumed via vcpkg and Conan. Optional backends SHALL be expressed as package
features/options that are off by default.

#### Scenario: Package features off by default
- WHEN the package is consumed via vcpkg or Conan with default features
- THEN no BLAS/LAPACK/GPU dependency is pulled in

### Requirement: Toolchain requirements

NumPP SHALL document and enforce minimum toolchain versions that fully support
the C++20 features it uses (clang, GCC, and Apple clang for the iOS path), and the
build SHALL fail configuration with a clear message on an unsupported compiler
rather than failing later with obscure errors.

#### Scenario: Unsupported compiler fails early
- WHEN the project is configured with a compiler below the documented minimum
- THEN configuration fails with a message naming the required minimum version

### Requirement: Symbol visibility and export macro

NumPP SHALL default to hidden symbol visibility and export its public API through
a generated export macro, so the shared library exposes only the intended
surface. Internal symbols SHALL NOT be exported.

#### Scenario: Only public symbols exported
- WHEN the shared library is built with default visibility settings
- THEN public API symbols are exported and internal-only symbols are not

### Requirement: iOS and Android build contract

NumPP SHALL be cross-compilable for iOS and Android with all backend feature
flags disabled, and CI SHALL include a job that performs these cross-compilations
to guard the portability promise. The mobile build SHALL produce a usable core
library.

#### Scenario: iOS cross-compile in CI
- WHEN CI cross-compiles the core for iOS with all backend flags OFF
- THEN the build succeeds and the resulting library exposes the core API

#### Scenario: Android cross-compile in CI
- WHEN CI cross-compiles the core for Android (NDK) with all backend flags OFF
- THEN the build succeeds and the resulting library exposes the core API
