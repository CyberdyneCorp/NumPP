// Minimal vendored DLPack ABI (https://github.com/dmlc/dlpack) — the header-only
// cross-framework tensor-exchange protocol. Only the structs/enums NumPP uses.
#pragma once

#include <cstdint>

extern "C" {

typedef enum { kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLOpenCL = 4 } DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;  // in elements (not bytes); NULL means C-contiguous
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

}  // extern "C"
