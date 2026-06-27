# dtype-system Specification

## Purpose
TBD - created by archiving change bootstrap-numpp-foundation. Update Purpose after archive.
## Requirements
### Requirement: Numeric dtype set

NumPP SHALL provide a `DType` value type identifying at least: `bool`,
`int8/16/32/64`, `uint8/16/32/64`, `float16`, `float32`, `float64`,
`complex64`, `complex128`. Each dtype SHALL expose its item size in bytes, a
kind code (`b`,`i`,`u`,`f`,`c`), and a stable name matching NumPy's
(`"int32"`, `"float64"`, `"complex128"`, ...).

#### Scenario: Item sizes
- WHEN the item sizes of `float64`, `complex128`, `int8`, `float16` are queried
- THEN they are `8`, `16`, `1`, `2` respectively

#### Scenario: Kind codes match NumPy
- WHEN the kind of `uint16` and `complex64` are queried
- THEN they are `'u'` and `'c'` respectively

### Requirement: dtype construction from name and aliases

NumPP SHALL construct a `DType` from its canonical NumPy name (`"int32"`,
`"float64"`, `"complex128"`, ...) and from the common aliases NumPy accepts
(e.g. `"float"`→float64, `"int"`→the default integer, `"bool"`, single-character
kind+size codes such as `"f8"`, `"i4"`, `"c16"`). An unrecognized name SHALL
raise a `type_error`.

#### Scenario: Construct from canonical name
- WHEN a `DType` is constructed from `"float32"`
- THEN it equals the `float32` dtype

#### Scenario: Construct from kind+size code
- WHEN a `DType` is constructed from `"i4"`
- THEN it equals the `int32` dtype

#### Scenario: Unknown name rejected
- WHEN a `DType` is constructed from `"float7"`
- THEN a `numpp::type_error` is raised

### Requirement: Default platform dtypes

NumPP SHALL define the default floating dtype as `float64`, the default complex
dtype as `complex128`, and the default integer dtype as a platform-dependent
signed integer matching NumPy's default on the same platform (typically `int64`
on 64-bit LP64 systems). These defaults SHALL be used when a routine is asked to
infer a dtype.

#### Scenario: Default float and integer
- WHEN `zeros([2])` is created with no dtype specified
- THEN its dtype is `float64`
- AND `arange(0, 3)` with no dtype specified has the default integer dtype

### Requirement: Native byte order only

NumPP SHALL store and operate on data in the host machine's native byte order.
Non-native byte order is a non-goal for this foundation; a dtype SHALL report its
byte order as native, and byte-swapped input handling is deferred to a later
change. This limitation SHALL be documented.

#### Scenario: dtype reports native order
- WHEN the byte order of any supported dtype is queried
- THEN it reports native (`'='`) order

### Requirement: Type promotion (result_type)

NumPP SHALL provide `result_type(a, b, ...)` returning the common dtype two or
more dtypes promote to, following NumPy's dtype-based promotion rules (NEP 50).
Promotion SHALL be associative and commutative and SHALL match NumPy's
`numpy.result_type` for all pairs of the supported numeric dtypes.

#### Scenario: Integer + float promotes to float
- WHEN `result_type(int32, float32)` is evaluated
- THEN the result is `float64` (matching NumPy's int32+float32 promotion)

#### Scenario: Mixed-width integers
- WHEN `result_type(int8, int64)` is evaluated
- THEN the result is `int64`

#### Scenario: Signed + unsigned
- WHEN `result_type(int16, uint16)` is evaluated
- THEN the result is `int32` (matching NumPy)

#### Scenario: Real + complex
- WHEN `result_type(float64, complex64)` is evaluated
- THEN the result is `complex128`

#### Scenario: Parity with NumPy across all pairs
- WHEN `result_type` is computed for every ordered pair of supported dtypes
- THEN each result equals `numpy.result_type` for the same pair (oracle-checked)

### Requirement: Casting rules

NumPP SHALL provide `can_cast(from, to, casting)` for casting modes `no`,
`equiv`, `safe`, `same_kind`, and `unsafe`, returning whether the cast is
permitted, matching NumPy's `numpy.can_cast`.

#### Scenario: Safe widening
- WHEN `can_cast(int16, int32, "safe")` is queried
- THEN the result is true

#### Scenario: Unsafe narrowing rejected under safe
- WHEN `can_cast(float64, int32, "safe")` is queried
- THEN the result is false

#### Scenario: same_kind allows within-kind narrowing
- WHEN `can_cast(float64, float32, "same_kind")` is queried
- THEN the result is true

### Requirement: Cast kernels

NumPP SHALL provide an `astype(array, dtype, casting=unsafe)` operation that
produces a new array whose elements are the source values converted to the
target dtype, for every supported (from, to) dtype pair, including
half<->float<->double and real<->complex conversions. The conversion SHALL be
performed by a strided kernel that works on non-contiguous inputs.

#### Scenario: Float to int truncates toward zero
- GIVEN a `float64` array `[1.9, -1.9, 2.5]`
- WHEN cast to `int32`
- THEN the result is `[1, -1, 2]` (matching NumPy)

#### Scenario: Real to complex
- GIVEN a `float32` array `[1.0, 2.0]`
- WHEN cast to `complex64`
- THEN the result is `[1+0j, 2+0j]`

#### Scenario: half round-trip
- GIVEN a `float16` array of representable values
- WHEN cast to `float32` and back to `float16`
- THEN the values are unchanged

#### Scenario: Cast respects strides
- GIVEN a non-contiguous `float64` view
- WHEN cast to `int32`
- THEN the result equals casting `numpy`'s equivalent non-contiguous view (oracle-checked)

### Requirement: Casting-rule helpers
NumPP SHALL provide `promote_types(a, b)` returning the smallest dtype both inputs
can safely cast to, and `min_scalar_type(value)` returning the smallest dtype that
can hold a scalar, matching numpy (with NEP-50 semantics already used by
`result_type`/`can_cast`/`common_type`).

#### Scenario: promotion and minimal scalar type match numpy
- WHEN `promote_types` is queried for representative dtype pairs (int/uint/float/complex/bool)
- THEN the result equals `numpy.promote_types`
- WHEN `min_scalar_type` is queried for representative integer and floating scalars
- THEN the result equals `numpy.min_scalar_type`

