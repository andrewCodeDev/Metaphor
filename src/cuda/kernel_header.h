#ifndef __TYPE_INDICATORS_H__
#define __TYPE_INDICATORS_H__

#include "tensor_types.h"

// This must be cast back to a
// CUstream before using it to
// launch cuda kernels
typedef struct {
    void* ptr;
} Stream;

// generator types
#define RScalar float
#define CScalar c32
#define RTensor RTensor32
#define CTensor CTensor32
#define WARP_SIZE 32

#if defined(__cplusplus)
////////////////////////////////
// NVCC COMPILER STUFF /////////
#include <cmath>
#include <stdio.h>
#include "../../dependencies/cuda/include/cuda.h"
#include "../../dependencies/cuda/include/cuda_runtime.h"
#include "../../dependencies/cuda/include/cooperative_groups.h"

#if defined(__CUDACC__)

  inline CUstream getStream(Stream stream) {
    return static_cast<CUstream>(stream.ptr);
  }

  namespace cg = cooperative_groups;
#endif

template<class T>
__device__ T ctanh(T x) { 
  auto a = rtanh(x.r);
  auto b = rtan(x.i);
  return cdiv(T{ .r = a, .i = b }, T{ .r = decltype(a){1.0}, .i = a * b });
}

#define CUDA_ASSERT(err) (HandleError( err, __FILE__, __LINE__ ))
inline void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CURESULT_ASSERT(err) (handleCUResultError( err, __FILE__, __LINE__ ))
inline void handleCUResultError(CUresult err, const char *file, int line)
{
  if (err != CUDA_SUCCESS) {
    const char** msg = nullptr;

    cuGetErrorString(err, msg);

    if (*msg) {
      printf("%s in %s at line %d\n", *msg, file, line);
    } else {
      printf("Unkown error in %s at line %d\n", file, line);
    }   
    exit(EXIT_FAILURE);
  }
}

#define GRID_1D(N) (((N) / 32) + 1)

// TODO: Move this math stuff to a different header
__device__ __inline__ r16 conjmul(c16 x) { return x.r * x.r + x.i * x.i; }
__device__ __inline__ r32 conjmul(c32 x) { return x.r * x.r + x.i * x.i; }
__device__ __inline__ r64 conjmul(c64 x) { return x.r * x.r + x.i * x.i; }

__device__ __inline__ r16 rtanh(r16 x) { return r16(std::tanh(static_cast<r32>(x))); }
__device__ __inline__ r32 rtanh(r32 x) { return std::tanh(x); }
__device__ __inline__ r64 rtanh(r64 x) { return std::tanh(x); }

__device__ __inline__ r16 rtan(r16 x) { return r16(std::tan(static_cast<r32>(x))); }
__device__ __inline__ r32 rtan(r32 x) { return std::tan(x); }
__device__ __inline__ r64 rtan(r64 x) { return std::tan(x); }

template<class T>
__device__ __inline__ T cdiv(T x, T y) { 
  auto u = conjmul(y); return T{ .r = x.r / u, .i = x.i / u };  
}
template<class T>
__device__ __inline__ T cmul(T x, T y) { 
  return T { .r = (x.r * y.r - x.i * y.i), .i = (x.r * y.i + x.i * y.r) };
}


#endif // nvcc stuff
#endif // header guard
