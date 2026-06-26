# Tasks — masked arrays extended

- [x] elementwise arithmetic: add, subtract, multiply, divide (mask union)
- [x] comparison/range constructors: masked_less/less_equal/greater_equal/not_equal/inside/outside/values
- [x] per-axis reductions: sum/prod/mean/max/min/count along an axis (slice all-masked -> masked)
- [x] accessors: getmask, getdata
- [x] NumPy-oracle tests vs np.ma (via .filled() + getmaskarray); clang + gcc + ASan/UBSan green
- [x] openspec validate; PR + merge + archive
