# error-handling Specification

## ADDED Requirements

### Requirement: Exception-based error model

NumPP SHALL report recoverable usage errors by throwing C++ exceptions derived
from a common base `numpp::error` (itself derived from `std::exception`). The
library SHALL NOT use the CPython C-API, `errno`-style global state, or process
termination for these conditions. Every exception SHALL carry a human-readable
message.

#### Scenario: Errors derive from a common base
- WHEN any NumPP operation raises an error
- THEN the thrown object is catchable as `numpp::error` and as `std::exception`
- AND `what()` returns a non-empty message

### Requirement: Typed exception hierarchy mirroring NumPy

NumPP SHALL provide exception types corresponding to the NumPy error categories
its operations can produce, at minimum: `value_error` (bad shapes/values),
`type_error` (incompatible dtypes/casting), `index_error` (out-of-range index),
`axis_error` (invalid axis), `linalg_error` (numerical failure such as singular
matrix), and `not_implemented_error` (unsupported combination). Each SHALL derive
from `numpp::error`.

#### Scenario: Shape mismatch raises value_error
- WHEN an operation receives non-broadcastable shapes
- THEN a `numpp::value_error` is thrown

#### Scenario: Out-of-range index raises index_error
- GIVEN an array of shape `[3]`
- WHEN element index `5` is requested
- THEN a `numpp::index_error` is thrown

#### Scenario: Invalid axis raises axis_error
- GIVEN an array of rank 2
- WHEN an operation is given axis `3`
- THEN a `numpp::axis_error` is thrown, and the message reports the valid axis range

### Requirement: Strong exception safety for mutating operations

Operations that fail by throwing SHALL leave their input arguments unmodified
(strong exception guarantee). A failed operation SHALL NOT leak the memory it had
begun to allocate.

#### Scenario: Failed reshape does not mutate input
- GIVEN an array `a`
- WHEN `reshape` is called with an incompatible element count and throws
- THEN `a`'s shape, strides, and data are unchanged

### Requirement: Domain and numerical conditions

NumPP SHALL define, per operation, whether an out-of-domain or non-finite input
produces an exception or a propagated non-finite result (`NaN`/`inf`). Element-
wise math SHALL follow IEEE-754 and propagate `NaN`/`inf` rather than throw;
structural and dimensional errors SHALL throw. This contract SHALL be documented
for each operation that can encounter such inputs.

#### Scenario: Non-finite propagates, not throws
- GIVEN a float array containing `inf` and `NaN`
- WHEN an element-wise operation is applied
- THEN the result contains the IEEE-754 propagated values and no exception is thrown

#### Scenario: Singular matrix raises linalg_error
- GIVEN a singular square matrix
- WHEN an operation requiring invertibility (e.g. `inv`) is applied
- THEN a `numpp::linalg_error` is thrown (this contract applies when linalg lands; defined here)
