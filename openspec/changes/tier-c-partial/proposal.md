# Tier C (achievable subset) — datetime business-day, numpy.testing, masked arrays

## Why
Tier C is the large/specialized tier. This change delivers the parts that are
genuinely implementable and oracle-testable in a portable, dependency-free C++
clean-room port; the rest of Tier C is documented as deferred with rationale.

## What changes
- **datetime-completion**: is_busday, busday_count, busday_offset, datetime_as_string
- **numpy-testing**: array_equal, array_equiv, assert_array_equal, assert_allclose,
  assert_array_almost_equal, assert_array_less
- **masked-arrays** (numpy.ma MVP): MaskedArray + masked_where/masked_invalid/
  masked_equal/masked_greater, filled, compressed, count, sum/mean/min/max

## Non-goals (Tier C items deferred, with rationale)
- **Real GPU backends** (Metal/CUDA/Vulkan/OpenCL): require platform SDKs and GPU
  hardware to build and oracle-test — not possible in this environment. The
  weak-vtable architecture + CPU-reference device already prove the dispatch path.
- **Bit-exact long-tail** #7 choice(replace=False), #8 ziggurat, #9 MT19937
  stream: deep reverse-engineering of numpy's exact internal algorithms (the same
  hard class as Philox #36; MT19937's init_by_array seeding + index offset was
  verified non-trivial). Tracked, not in this slice.
- **object dtype** (type-erased storage) and **DEFLATE-compressed npz**: large
  additions conflicting with the no-dependency portability goal. Structured-field
  access already exists; npz is already numpy-readable uncompressed.
- masked-array arithmetic operators, per-axis masked reductions, hard/soft masks.
