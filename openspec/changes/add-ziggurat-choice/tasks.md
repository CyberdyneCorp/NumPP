# Tasks — ziggurat + choice bit-exactness

- [x] extract numpy ziggurat tables + constants into src/random/ziggurat_constants.h
- [x] bit-exact standard_normal (ziggurat) -> normal also bit-exact
- [x] bit-exact standard_exponential (ziggurat)
- [x] bit-exact choice(replace=False): Floyd hash-set + tail-shuffle branch (Lemire bounded)
- [x] bit-exact oracle tests (standard_normal/exponential/normal/choice both paths); clang + ASan/UBSan green
- [x] close #8 (ziggurat) and #7 (choice replace=False)
- [x] openspec validate; PR + merge + archive
