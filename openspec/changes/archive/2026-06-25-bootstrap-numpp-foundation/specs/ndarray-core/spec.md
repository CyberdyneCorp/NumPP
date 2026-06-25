# ndarray-core Specification

## ADDED Requirements

### Requirement: N-dimensional strided array container

NumPP SHALL provide `numpp::ndarray`, an N-dimensional array described by a
shared data buffer, a dtype, a shape (list of non-negative dimension sizes),
byte strides (one per dimension, may be zero or negative), a byte offset into
the buffer, and flags. Rank 0 (scalar) through at least rank 32 SHALL be
supported. Element access at logical index `(i0..ik)` SHALL read the buffer at
`offset + sum(ij * stride_j)`.

#### Scenario: Construct from shape and dtype
- WHEN an `ndarray` is created with shape `[2, 3]` and dtype `float64`
- THEN `ndim()` is 2, `shape()` is `[2, 3]`, `size()` is 6, `itemsize()` is 8
- AND the array is C-contiguous with strides `[24, 8]` (bytes)

#### Scenario: Zero-dimensional array
- WHEN an `ndarray` is created with shape `[]`
- THEN `ndim()` is 0 and `size()` is 1

### Requirement: Buffer ownership and view aliasing

Multiple arrays SHALL be able to share one reference-counted data buffer. The
buffer SHALL be released only when the last referencing array is destroyed.
Operations that produce a view SHALL share the source buffer without copying
data; mutating a writeable view SHALL be observable through the source array.

#### Scenario: View shares storage
- GIVEN an array `a` of shape `[4]`
- WHEN `b` is a view of `a` (e.g. a slice or transpose)
- AND an element of `b` is assigned a new value
- THEN reading the corresponding element of `a` returns the new value

#### Scenario: Lifetime via shared buffer
- GIVEN a view `b` of array `a`
- WHEN `a` is destroyed while `b` is still alive
- THEN `b` remains valid and its data is intact

### Requirement: Contiguity flags

NumPP SHALL track `C_CONTIGUOUS` and `F_CONTIGUOUS` flags and expose them.
A freshly allocated array SHALL be C-contiguous. The flags SHALL be recomputed
for views so they reflect the actual stride layout.

#### Scenario: Transpose toggles contiguity
- GIVEN a C-contiguous 2-D array `a`
- WHEN `b = a.transpose()`
- THEN `b` is F-contiguous and not C-contiguous (for shapes where these differ)
- AND `b` shares `a`'s buffer

### Requirement: Reshape

NumPP SHALL provide `reshape(new_shape)` matching NumPy semantics: the total
number of elements MUST be preserved, exactly one dimension MAY be `-1` and is
inferred. Reshape SHALL return a view when the layout permits (e.g. contiguous
arrays) and SHALL copy otherwise. An incompatible element count SHALL raise an
error.

#### Scenario: Compatible reshape returns a view
- GIVEN a C-contiguous array of shape `[2, 6]`
- WHEN `reshape([3, 4])` is called
- THEN the result has shape `[3, 4]` and shares the source buffer

#### Scenario: Inferred dimension
- GIVEN an array of size 12
- WHEN `reshape([3, -1])` is called
- THEN the result has shape `[3, 4]`

#### Scenario: Incompatible reshape
- GIVEN an array of size 12
- WHEN `reshape([5, 5])` is called
- THEN an error SHALL be raised

### Requirement: Transpose and axis permutation

NumPP SHALL provide `transpose()` (reverse all axes) and `transpose(axes)`
(arbitrary permutation) as zero-copy stride/shape permutations, plus `swapaxes`
and `squeeze` (drop size-1 axes). Results SHALL share the source buffer.

#### Scenario: Permute axes
- GIVEN an array of shape `[2, 3, 4]`
- WHEN `transpose([2, 0, 1])` is called
- THEN the result has shape `[4, 2, 3]` and shares the buffer

#### Scenario: Squeeze
- GIVEN an array of shape `[1, 3, 1]`
- WHEN `squeeze()` is called
- THEN the result has shape `[3]`

### Requirement: Basic and sliced indexing

NumPP SHALL support integer indexing (returning a lower-rank view), and slice
indexing with start/stop/step including negative indices and negative steps,
matching NumPy's basic-indexing semantics. Basic indexing SHALL return a view.

#### Scenario: Integer index reduces rank
- GIVEN an array `a` of shape `[3, 4]`
- WHEN `a[1]` is taken
- THEN the result is a view of shape `[4]` aliasing row 1 of `a`

#### Scenario: Strided slice
- GIVEN a 1-D array `[0,1,2,3,4,5]`
- WHEN sliced with `start=1, stop=5, step=2`
- THEN the result is a view equal to `[1, 3]`

#### Scenario: Reversing slice
- GIVEN a 1-D array `[0,1,2,3]`
- WHEN sliced with `step=-1`
- THEN the result is a view equal to `[3, 2, 1, 0]` (negative stride)

### Requirement: Copy and contiguous materialization

NumPP SHALL provide `copy(order=C|F)` that returns an owning array with freshly
allocated, contiguous storage independent of the source, and `ascontiguousarray`
/ `asfortranarray` that return the input unchanged when it already satisfies the
requested layout and a contiguous copy otherwise.

#### Scenario: Copy is independent
- GIVEN an array `a`
- WHEN `b = a.copy()` and an element of `b` is modified
- THEN the corresponding element of `a` is unchanged

#### Scenario: Already-contiguous fast path
- GIVEN a C-contiguous array `a`
- WHEN `ascontiguousarray(a)` is called
- THEN the result shares `a`'s buffer (no copy)

### Requirement: Broadcasting

NumPP SHALL compute broadcast shapes using NumPy rules: shapes are aligned on
the trailing dimension; a dimension of size 1 is stretched to match; a missing
leading dimension is treated as size 1; otherwise dimensions must be equal.
`broadcast_to(array, shape)` SHALL return a zero-copy view with size-1 (and
prepended) dimensions given stride 0. Incompatible shapes SHALL raise an error.

#### Scenario: Stretch a size-1 dimension
- GIVEN shapes `[3, 1]` and `[1, 4]`
- WHEN broadcast together
- THEN the result shape is `[3, 4]`

#### Scenario: Prepend dimensions
- GIVEN shapes `[5, 4]` and `[4]`
- WHEN broadcast together
- THEN the result shape is `[5, 4]`

#### Scenario: broadcast_to is a zero-stride view
- GIVEN an array of shape `[3, 1]`
- WHEN `broadcast_to(a, [3, 4])` is called
- THEN the result has shape `[3, 4]`, shares the buffer, and has stride 0 on the last axis

#### Scenario: Incompatible shapes
- GIVEN shapes `[3]` and `[4]`
- WHEN broadcast together
- THEN an error SHALL be raised

### Requirement: Array creation routines

NumPP SHALL provide the core array-creation routines matching NumPy semantics:
`empty(shape, dtype)`, `zeros`, `ones`, `full(shape, value, dtype)`,
`empty_like/zeros_like/ones_like/full_like`, `eye(n, m=n, k=0, dtype)`,
`identity(n, dtype)`, `arange(start, stop, step, dtype)`, and
`linspace(start, stop, num, endpoint=true, dtype)`. Each SHALL produce a
C-contiguous owning array of the requested dtype.

#### Scenario: zeros and ones
- WHEN `zeros([2, 2], float64)` and `ones([2], int32)` are created
- THEN the first is all `0.0` with dtype `float64` and the second is `[1, 1]` with dtype `int32`

#### Scenario: arange
- WHEN `arange(0, 10, 2, int64)` is created
- THEN the result is `[0, 2, 4, 6, 8]`

#### Scenario: linspace endpoints
- WHEN `linspace(0.0, 1.0, 5)` is created
- THEN the result is `[0.0, 0.25, 0.5, 0.75, 1.0]` (matching NumPy)

#### Scenario: eye with offset diagonal
- WHEN `eye(3, k=1)` is created
- THEN the superdiagonal is `1` and all other entries are `0`

#### Scenario: like-constructors copy shape and dtype
- GIVEN an array `a` of shape `[2, 3]` and dtype `float32`
- WHEN `zeros_like(a)` is created
- THEN the result has shape `[2, 3]`, dtype `float32`, and all elements `0`

### Requirement: Flatten and ravel

NumPP SHALL provide `ravel(order=C|F)` returning a 1-D array that is a view when
the requested order permits and a copy otherwise, and `flatten(order=C|F)` which
always returns an owning 1-D copy. The element order SHALL match NumPy.

#### Scenario: ravel of contiguous array is a view
- GIVEN a C-contiguous array of shape `[2, 3]`
- WHEN `ravel()` (C order) is called
- THEN the result is a 1-D view sharing the buffer

#### Scenario: flatten always copies
- GIVEN any array `a`
- WHEN `flatten()` is called and an element of the result is modified
- THEN `a` is unchanged

#### Scenario: ravel of non-contiguous order copies
- GIVEN a transposed (F-contiguous) view
- WHEN `ravel()` is called in C order
- THEN the result is a 1-D copy whose values equal NumPy's `a.ravel(order='C')`

### Requirement: Dimension insertion

NumPP SHALL provide `expand_dims(array, axis)` and newaxis-style indexing that
insert size-1 dimensions, returning a zero-copy view with the inserted axis given
the appropriate stride. Negative axis values SHALL be accepted.

#### Scenario: expand_dims
- GIVEN an array of shape `[3, 4]`
- WHEN `expand_dims(a, 1)` is called
- THEN the result has shape `[3, 1, 4]` and shares the buffer

#### Scenario: expand_dims at the end
- GIVEN an array of shape `[3]`
- WHEN `expand_dims(a, -1)` is called
- THEN the result has shape `[3, 1]`

### Requirement: Writeability and read-only views

NumPP SHALL track a `WRITEABLE` flag per array. A view derived from a read-only
array or a broadcasted (zero-stride) array SHALL be read-only. Attempting to
assign through a read-only array SHALL raise an error. The flag SHALL be
queryable.

#### Scenario: Broadcasted view is read-only
- GIVEN a zero-stride view produced by `broadcast_to`
- WHEN an element assignment is attempted through it
- THEN an error is raised and no data is modified

#### Scenario: Writeability propagates to views
- GIVEN a read-only array `a`
- WHEN a slice view `b` of `a` is taken
- THEN `b` is also read-only

### Requirement: Raw data access and fill

NumPP SHALL expose typed access to the underlying buffer for callers that have
established the dtype: a raw data pointer with the array's byte offset/strides, a
scalar `item(index...)` accessor that returns a single element, and a
`fill(value)` operation that sets every element of a writeable array. Access that
violates the array's dtype or writeability SHALL be rejected.

#### Scenario: fill sets all elements
- GIVEN a writeable `float64` array of shape `[2, 2]`
- WHEN `fill(3.5)` is called
- THEN every element equals `3.5`

#### Scenario: item reads a single element
- GIVEN an array `a` with `a[1, 2] == 7`
- WHEN `item(1, 2)` is called
- THEN it returns `7`

#### Scenario: fill on read-only array is rejected
- GIVEN a read-only array
- WHEN `fill` is attempted
- THEN an error is raised

### Requirement: Element iteration order

NumPP SHALL provide iteration that visits elements in a well-defined order
respecting strides, including for non-contiguous and broadcasted (zero-stride)
views, so that kernels can traverse any array without first copying it to
contiguous storage.

#### Scenario: Iterate a transposed view
- GIVEN a C-contiguous array of shape `[2, 3]` and its transpose `t` of shape `[3, 2]`
- WHEN `t` is iterated in C order
- THEN the visited values equal NumPy's `a.T.ravel(order='C')`
