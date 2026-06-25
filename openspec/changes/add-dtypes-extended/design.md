# Design — extended dtypes

## Core DType extension (the invasive part)

`DType` was a fixed 14-value enum with a static itemsize table. To carry the
extra information extended dtypes need (string width, datetime unit, struct
fields) without disturbing the numeric dtypes, DType gains an optional
`std::shared_ptr<const DTypeMeta>` side-channel that is **null for all 14 numeric
dtypes** — so numeric DType copies/compares stay cheap (null shared_ptr copy
touches no refcount) and all 887 pre-existing oracle checks remain green.

- New `DTypeId` values String/Bytes/Datetime64/Timedelta64/Struct sit *after* the
  numeric range; `kNumDTypes` (14) still bounds the promotion/casting tables, and
  `result_type`/`can_cast` guard `is_extended()` (extended types only match
  themselves — no numeric<->extended promotion yet).
- `itemsize()` now returns `int64_t` (strings exceed 255 bytes) and consults the
  metadata; `kind()`/`name()` switch on the extended ids.
- `visit_dtype` throws for extended ids (string/struct ops never go through it).

Tradeoff: DType is no longer `constexpr`/trivially-copyable. Verified no code
required that. Documented here per the autonomous-loop guidance.

## Strings
'U' = UTF-32 (4 bytes/char), 'S' = bytes; fixed width = itemsize. Element access
encodes/decodes UTF-8<->UTF-32. Printing quotes elements; repr emits dtype='<U..'.
.npy descr maps to '<U{n}'/'|S{n}'.
