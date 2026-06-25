# ufuncs Specification

## ADDED Requirements

### Requirement: Element-wise binary ufuncs with broadcasting

NumPP SHALL provide binary ufuncs that broadcast their inputs per NumPy rules,
resolve the output dtype per NumPy's type rules, and compute element-wise:
arithmetic (`add`, `subtract`, `multiply`, `divide`/`true_divide`,
`floor_divide`, `mod`, `power`), with results matching NumPy.

#### Scenario: Broadcasted addition
- GIVEN arrays of shape `[3, 1]` and `[1, 4]`
- WHEN `add` is applied
- THEN the result has shape `[3, 4]` and equals `numpy.add` of the same inputs

#### Scenario: True division promotes integers to float
- WHEN `divide` is applied to two `int32` arrays
- THEN the result dtype is `float64` and values match `numpy.true_divide`

### Requirement: Comparison and logical ufuncs

NumPP SHALL provide comparison ufuncs (`equal`, `not_equal`, `less`,
`less_equal`, `greater`, `greater_equal`) and logical ufuncs (`logical_and`,
`logical_or`, `logical_xor`, `logical_not`) returning `bool` arrays matching
NumPy.

#### Scenario: Element-wise comparison returns bool
- WHEN `less` is applied to two arrays
- THEN the result dtype is `bool` and equals `numpy.less` of the same inputs

### Requirement: Unary math ufuncs

NumPP SHALL provide unary ufuncs — `negative`, `absolute`, `sqrt`, `exp`, `log`,
`sin`, `cos`, `tan`, `floor`, `ceil`, and others — promoting integer inputs to
floating point for transcendental functions, matching NumPy's output dtype and
values within floating-point tolerance.

#### Scenario: sqrt of integers yields float64
- WHEN `sqrt` is applied to an `int64` array
- THEN the result dtype is `float64` and values match `numpy.sqrt`

#### Scenario: Non-finite propagation
- WHEN `log` is applied to an array containing `0` and a negative value
- THEN the result contains `-inf` and `nan` as NumPy produces, without throwing

### Requirement: Reductions

NumPP SHALL provide reductions `sum`, `prod`, `min`, `max`, `mean`, `std`,
`var`, `any`, `all` supporting `axis` (int or none), `keepdims`, and (where
applicable) `dtype`, matching NumPy.

#### Scenario: Sum over an axis
- GIVEN an array of shape `[2, 3]`
- WHEN `sum(axis=0)` is applied
- THEN the result has shape `[3]` and equals `numpy.sum(a, axis=0)`

#### Scenario: keepdims
- WHEN `sum(axis=1, keepdims=true)` is applied to a `[2, 3]` array
- THEN the result has shape `[2, 1]`

### Requirement: Operator overloads

NumPP SHALL overload `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `<=`, `>`, `>=` on
`ndarray` to delegate to the corresponding ufuncs.

#### Scenario: Operator delegates to ufunc
- WHEN `a + b` is evaluated for two arrays
- THEN the result equals `add(a, b)`
