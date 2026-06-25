# oracle-testing Specification

## ADDED Requirements

### Requirement: NumPy validation oracle

NumPP's test suite SHALL be able to validate results against real Python NumPy:
a test helper SHALL evaluate a reference computation in NumPy and the test SHALL
assert that the NumPP result matches the NumPy result element-wise within a
specified relative and absolute tolerance (`allclose`), accounting for dtype and
shape.

#### Scenario: Compare against NumPy
- GIVEN a NumPP computation and the equivalent NumPy expression
- WHEN both are evaluated on the same inputs
- THEN the NumPP result and the NumPy result satisfy `allclose(rtol, atol)`

#### Scenario: Shape and dtype are checked
- WHEN a NumPP result is compared to its NumPy oracle
- THEN the comparison fails if shapes differ or if the result dtype does not match the expected dtype

### Requirement: Tolerance control

The oracle comparison SHALL allow per-test relative and absolute tolerances, and
SHALL support exact (bitwise/integer) comparison for integer and boolean dtypes
where NumPy produces exact results.

#### Scenario: Exact integer comparison
- GIVEN an integer-dtype NumPP result and its NumPy oracle
- WHEN compared in exact mode
- THEN every element must be identical

### Requirement: Non-finite comparison semantics

The oracle comparison SHALL handle IEEE-754 non-finite values: `inf` SHALL match
`inf` of the same sign, and `NaN` SHALL be treated as matching `NaN` when an
`equal_nan` option is enabled (matching NumPy's `allclose(equal_nan=...)`). The
default SHALL match NumPy's default.

#### Scenario: NaN matches NaN under equal_nan
- GIVEN a NumPP result and oracle that both contain `NaN` in the same positions
- WHEN compared with `equal_nan` enabled
- THEN the comparison passes

#### Scenario: Signed infinity must match
- GIVEN a result with `+inf` where the oracle has `-inf`
- WHEN compared
- THEN the comparison fails

### Requirement: Deterministic test inputs

Test inputs generated for oracle comparison SHALL be reproducible: any
pseudo-random input SHALL be produced from a fixed, recorded seed so that the same
case yields the same inputs across runs and matches the inputs used to produce
any frozen golden data.

#### Scenario: Seeded inputs are reproducible
- WHEN a randomized oracle test case is generated twice with the same seed
- THEN both runs produce identical input arrays

### Requirement: Frozen golden mode for hermetic CI

NumPP SHALL provide a generation mode that serializes oracle arrays produced by
NumPy into checked-in golden files, and a comparison mode that runs the test
suite against those golden files without invoking Python or requiring NumPy to be
installed. CI SHALL be able to run in frozen mode.

#### Scenario: Generate golden data
- WHEN the oracle generator is run with Python and NumPy available
- THEN it writes golden result files for the registered test cases

#### Scenario: Run without Python
- GIVEN previously generated golden files
- WHEN the test suite is run in frozen mode on a machine without Python/NumPy
- THEN tests load the golden data and pass without invoking Python

### Requirement: Oracle availability is optional at build time

The presence of Python/NumPy SHALL NOT be a hard requirement to build NumPP or to
run the frozen test subset. Tests that require a live NumPy oracle SHALL be
skipped (not failed) when Python/NumPy is unavailable and no golden data exists.

#### Scenario: Missing oracle skips live tests
- GIVEN a machine without Python or NumPy and no golden data for a live-only test
- WHEN the test suite runs
- THEN the live-oracle tests are reported as skipped, not failed
