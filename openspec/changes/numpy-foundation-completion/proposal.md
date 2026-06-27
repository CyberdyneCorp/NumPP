# Complete the NumPy foundation (for a future C++ SciPy port)

## Why
NumPP is meant to be the **NumPy-equivalent foundation** that a separate C++ SciPy
port ("SciPP") will build on. The boundary is deliberate: anything in the
`numpy.*` namespace belongs in NumPP; anything in `scipy.*` (sparse, ODE/PDE
solvers, quadrature beyond `trapz`, optimization, interpolation beyond `interp`,
spatial/FEM) belongs in SciPP. To keep NumPP from diverging, this change closes
the remaining genuine `numpy.*` gaps a SciPP would stand on — and nothing more.

A focused spike confirmed the largest gap: NumPP's linear algebra is strictly
2-D. `det`/`inv`/`solve`/`cholesky`/`eig*`/`svd*`/`matrix_rank` all throw on a
stacked `(k, n, n)` input, whereas NumPy operates over the last two axes of an
N-D stack. Downstream array code (and a SciPP's own internals) assume this.

## What changes
### Tier 1 — pervasive, cheap, pure `numpy.*`
- **machine-limits**: `finfo` / `iinfo` (eps, tiny, max, min, resolution, bits).
- **ufunc-completion**: `isclose` (elementwise), `isposinf`, `isneginf`.
- **statistics**: `trapz` / `trapezoid` (the one integrator that lives in `numpy.*`).
- **dtype-system**: `promote_types`, `min_scalar_type` (complete the casting set;
  `result_type`/`can_cast`/`common_type` already exist).

### Tier 2 — completeness of existing features
- **linalg**: **batched/stacked** operation over the last two axes for
  `solve`/`inv`/`det`/`slogdet`/`matrix_power`/`cholesky`/`qr`/`eig`/`eigh`/
  `eigvals`/`eigvalsh`/`svd`/`svdvals`/`pinv`/`matrix_rank`/`lstsq`; plus
  `tensorsolve` / `tensorinv` (currently absent).
- **einsum**: ellipsis (`...`) broadcasting subscripts.
- **signal-poly**: `interp` `left` / `right` / `period` arguments.

## Non-goals (would diverge — these belong in SciPP or are out of charter)
- `scipy.*` capabilities: sparse matrices/solvers, ODE/PDE integrators, quadrature
  beyond `trapz`, optimization/root-finding, splines/N-D interpolation, spatial/mesh,
  special functions. These are SciPP's job and must NOT enter NumPP.
- Bucket A (`object` dtype, `recarray`, `frompyfunc`/multi-arg `vectorize`,
  `ctypeslib`, `np.matrix`): a C++ SciPP uses templates/variants/`std::function`,
  not Python-object arrays — adding them is pure divergence.
- Bucket B (external deps): zlib `savez_compressed`, FFTW, `longdouble`.
- Metal backend: wanted for the iPad target, but it is an acceleration backend
  behind the existing weak-vtable, not a `numpy.*` feature — tracked separately.
