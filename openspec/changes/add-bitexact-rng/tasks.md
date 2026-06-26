# Tasks — bit-exact Philox + MT19937

- [x] reverse-engineer numpy Philox: M0 = 0xD2E7470EE14C6C93, increment-before-generate
- [x] fix Philox to be bit-exact with numpy.random.Philox
- [x] add MT19937BitGen bit-exact with numpy.random.MT19937 (generate_state(624), mt[0]=0x80000000, pos=623)
- [x] bit-exact oracle tests for Philox and MT19937 (uint64 exact compare); clang + ASan/UBSan green
- [x] close #36 (Philox) and #9 (MT19937)
- [x] openspec validate; PR + merge + archive
