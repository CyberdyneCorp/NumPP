# gpu-kernels Specification

## Purpose
TBD - created by archiving change add-gpu-kernels. Update Purpose after archive.
## Requirements
### Requirement: GPU vtable dispatch for ufuncs and reductions
NumPP SHALL route a representative set of element-wise ufuncs (add, subtract,
multiply, divide, negative, sqrt, exp) and reductions (sum, prod) through an
optional weak-linked GpuVTable when one is registered, the operands are
contiguous and float/complex, and the size is at least NUMPP_GPU_MIN; otherwise
it SHALL use the portable CPU kernel. The chosen backend SHALL be reported by
last_backend().

#### Scenario: Device path equals CPU path
- GIVEN a build with a registered device backend
- WHEN an eligible op runs above the size threshold
- THEN the result equals the CPU kernel's result and last_backend() is Device

#### Scenario: Fallback below threshold and for unsupported dtypes
- WHEN an op runs below the threshold, or on an integer dtype
- THEN the CPU kernel is used and last_backend() is Cpu

#### Scenario: Default build unchanged
- GIVEN a build with no device backend (the default)
- WHEN any op runs
- THEN the CPU kernel is always used and all results are unchanged

