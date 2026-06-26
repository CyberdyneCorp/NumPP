# Interop: DLPack and memory-mapped arrays

## Why
Last item of the numpy-parity-roadmap backlog. Delivers the dependency-light,
portable parts of interop: the DLPack tensor-exchange protocol (cross-framework
zero-copy interchange) and memory-mapped arrays.

## What changes
- **interop** capability (numpp::interop):
  - **DLPack**: `to_dlpack(a)` exports a contiguous array as a `DLManagedTensor`
    (CPU device), keeping the source buffer alive via the managed-tensor deleter;
    `from_dlpack(t)` adopts a `DLManagedTensor` zero-copy and unmaps it via the
    array buffer's release. A minimal DLPack ABI is vendored (`interop/dlpack.h`).
  - **memmap**: `memmap(path, dtype, shape, mode)` returns an ndarray view backed
    by an `mmap`'d file (modes r / r+ / w+, MAP_SHARED); the region is unmapped
    when the last view is released.

## Non-goals
- **DEFLATE-compressed `savez_compressed`** (needs zlib), an **FFTW** FFT backend
  and **ctypeslib** (Python-specific) — all conflict with the no-dependency,
  iOS/Android-portable goal, so they stay deferred. `savez_compressed` already
  writes numpy-readable (uncompressed) npz today.
