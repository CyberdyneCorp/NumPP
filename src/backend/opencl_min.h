// Minimal vendored OpenCL C declarations so the OpenCL backend builds without
// the system CL/cl.h dev headers. We link the ICD loader (libOpenCL.so.1)
// directly. Only the subset of the OpenCL 1.2 API used by opencl_backend.cpp is
// declared here.
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_bool;
typedef cl_uint cl_device_info;
typedef cl_uint cl_program_build_info;

cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*);
cl_context clCreateContext(const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void*, void*);
cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void*, void*);
cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
cl_int clFinish(cl_command_queue);
cl_int clReleaseMemObject(cl_mem);

}  // extern "C"

#define NUMPP_CL_SUCCESS 0
#define NUMPP_CL_DEVICE_TYPE_GPU 4
#define NUMPP_CL_DEVICE_NAME 0x102B
#define NUMPP_CL_DEVICE_EXTENSIONS 0x1030
#define NUMPP_CL_MEM_READ_ONLY 4
#define NUMPP_CL_MEM_WRITE_ONLY 2
#define NUMPP_CL_MEM_READ_WRITE 1
#define NUMPP_CL_MEM_COPY_HOST_PTR 32
#define NUMPP_CL_TRUE 1
#define NUMPP_CL_PROGRAM_BUILD_LOG 0x1183
