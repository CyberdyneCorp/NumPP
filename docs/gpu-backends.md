# GPU backends

NumPP routes a representative set of element-wise ufuncs and reductions through
an optional, weak-linked `GpuVTable` when one is registered, the operands are
contiguous float/complex, and the size is at least `NUMPP_GPU_MIN` (default
65536). Otherwise the portable CPU kernel is always used. The active backend is
reported by `numpp::last_backend()` (`Backend::Cpu` / `Backend::Device`).

Three backends plug into the same vtable slot:

| Backend | Flag | Notes |
|---------|------|-------|
| null (default) | — | no device; always CPU |
| CPU-reference device | `-DNUMPP_WITH_REFGPU=ON` | runs device math on the host; proves the dispatch path |
| **OpenCL** | `-DNUMPP_WITH_OPENCL=ON` | **real GPU**: float32/float64 add/sub/mul/div, negate, sqrt, sum/prod |
| **CUDA** | `-DNUMPP_WITH_CUDA=ON` | **real NVIDIA GPU**: same op set via CUDA kernels |

## OpenCL backend

Validated on an NVIDIA GeForce RTX 5060 (fp64 supported). Arithmetic and `sqrt`
are IEEE-exact, so device results equal the CPU path bitwise; `exp`, complex and
integer dtypes decline to the CPU kernel.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNUMPP_WITH_OPENCL=ON
cmake --build build -j
```

Requirements: an OpenCL ICD loader (`libOpenCL.so.1`) and a GPU ICD (e.g. NVIDIA's
`libnvidia-opencl.so.1`, registered in `/etc/OpenCL/vendors/`). The build does
**not** need the system OpenCL dev headers — minimal declarations are vendored in
`src/backend/opencl_min.h`. If no OpenCL GPU is present, `gpu_vtable()` is null
and everything falls back to the CPU (so the OpenCL build is safe on headless
hosts).

Optional system OpenCL headers (not required):

```bash
sudo apt-get install -y opencl-headers ocl-icd-opencl-dev
```

## CUDA backend (requires the CUDA toolkit)

A CUDA backend follows the same vtable shape but needs `nvcc` + `cudart`, which
are not installed by default. For a Blackwell GPU (RTX 5060, sm_120) use CUDA
≥ 12.8 from NVIDIA's repo:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

(Or, simpler: `sudo apt-get install -y nvidia-cuda-toolkit` — this installs CUDA 12.0,
whose `nvcc` targets only up to sm_90. That still works on an sm_120 GPU via PTX JIT,
see below.)

### Building the CUDA backend

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNUMPP_WITH_CUDA=ON
cmake --build build -j
```

Implemented in `backend/cuda_backend.cu` (float32/float64 add/sub/mul/div, negate,
sqrt, sum/prod). Validated on an NVIDIA RTX 5060.

**Forward compatibility (older nvcc, newer GPU):** the RTX 5060 is Blackwell
(sm_120), but Ubuntu's `nvcc` is CUDA 12.0 (targets ≤ sm_90). The backend is
compiled to a **PTX-virtual architecture** (`CUDA_ARCHITECTURES "70-virtual"`),
so the driver JIT-compiles the PTX to the GPU's native sm_120 at runtime — no
toolkit upgrade required. Installing CUDA ≥ 12.8 would instead allow native SASS
for sm_120 (faster startup, no JIT). If no NVIDIA GPU is present, `gpu_vtable()`
is null and everything falls back to the CPU.
