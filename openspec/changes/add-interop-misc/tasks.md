# Tasks — interop (DLPack + memmap)

- [x] vendor minimal DLPack ABI (interop/dlpack.h)
- [x] to_dlpack / from_dlpack (zero-copy; managed-tensor + buffer-release lifetimes)
- [x] memmap(path, dtype, shape, mode) over mmap (r / r+ / w+, MAP_SHARED)
- [x] tests: DLPack round-trip + field checks; memmap write/read + numpy tofile interop
- [x] clang + gcc + ASan/UBSan green (no leaks/UAF in the mmap + deleter chain)
- [x] openspec validate; PR + merge + archive
