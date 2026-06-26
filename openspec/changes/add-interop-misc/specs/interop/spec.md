# interop Specification

## ADDED Requirements

### Requirement: DLPack tensor exchange
NumPP SHALL provide to_dlpack and from_dlpack implementing the DLPack protocol
(CPU device) for zero-copy array interchange, mapping NumPP dtypes to/from
DLDataType and preserving shape and (element) strides; lifetimes are managed by
the DLManagedTensor deleter and the array buffer's release.

#### Scenario: DLPack round-trips an array
- WHEN an array is exported with to_dlpack and re-imported with from_dlpack
- THEN the result equals the original (data, shape, dtype), and the DLTensor
  fields (ndim, dtype code/bits, shape, element strides) describe it correctly

### Requirement: Memory-mapped arrays
NumPP SHALL provide memmap returning an ndarray backed by an mmap'd file (modes
r / r+ / w+), with writes (r+/w+) shared to the file and the mapping released
with the last view.

#### Scenario: memmap round-trips through a file
- WHEN data is written through a w+ mapping and read back through an r mapping
- THEN the values match; and a file written by numpy.ndarray.tofile is read
  identically by memmap
