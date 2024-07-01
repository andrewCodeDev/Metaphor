#ifndef __TYPE_INDICATORS_H__
#define __TYPE_INDICATORS_H__

#include "tensor_types.h"

// This prevents a weird stack related
// bug. I assume this is because it prevents
// the compiler from loading these into
// 64-bit registers. 
typedef struct {
  void* ptr;
} PaddedPtr;

typedef struct {
  PaddedPtr cuda_stream;
  PaddedPtr blas_handle;
} StreamContext;

// Use context pointers to prevent slicing
// the StreamContext between registers and
// the stack. Corrupts Cuda context otherwise.

// generator types
#define Scalar float
#define SortPair_Scalar SortPair_r32

#define DIMPAD(M, N) (((M) + ((N) - 1)) / (N))

#if defined(__cplusplus)
////////////////////////////////
// NVCC COMPILER STUFF /////////
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <numeric>
#include <limits>
#include "../../deps/cuda/include/cuda.h"
#include "../../deps/cuda/include/cuda_runtime.h"
#include "../../deps/cuda/include/cooperative_groups.h"
#include "../../deps/cuda/include/cublas_v2.h"

// TODO: may not be portable
inline uint32_t nextPow2(uint32_t n) {
  return (n == 1) ? 1 : 1 << (32 - __builtin_clzl(n - 1)); 
}

// TODO: may not be portable
inline uint64_t nextPow2(uint64_t n) {
  return (n == 1) ? 1 : 1 << (64 - __builtin_clzl(n - 1)); 
}

#if defined(__CUDACC__)

inline CUstream get_stream(StreamContext context) {
  return static_cast<CUstream>(context.cuda_stream.ptr);
}

inline cublasHandle_t get_handle(StreamContext context) {
  return static_cast<cublasHandle_t>(context.blas_handle.ptr);
}

namespace cg = cooperative_groups;

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

#define CUBLAS_ASSERT(err) (handleCUResultError( err, __FILE__, __LINE__ ))
inline void handleCUResultError(cublasStatus_t err, const char *file, int line)
{
  // TODO: Report better cublas errors
  if (err != CUBLAS_STATUS_SUCCESS) {
      printf("Cublas failure in %s at line %d\n", file, line);
    exit(EXIT_FAILURE);
  }
}

#define GRID_1D(N) (((N) / 32) + 1)

#define DIVISIBLE_BY_FOUR(N) ((N & 3) == 0)

///////////////////////////////////////////////////
// TODO: Move this math stuff to a different header

__device__ __inline__ r16 rtanh(r16 x) { return r16(std::tanh(static_cast<r32>(x))); }
__device__ __inline__ r32 rtanh(r32 x) { return std::tanh(x); }
__device__ __inline__ r64 rtanh(r64 x) { return std::tanh(x); }

__device__ __inline__ r16 rtan(r16 x) { return r16(std::tan(static_cast<r32>(x))); }
__device__ __inline__ r32 rtan(r32 x) { return std::tan(x); }
__device__ __inline__ r64 rtan(r64 x) { return std::tan(x); }

__device__ __inline__ r16 rexp(r16 x) { return hexp(x); }
__device__ __inline__ r32 rexp(r32 x) { return std::exp(x); }
__device__ __inline__ r64 rexp(r64 x) { return std::exp(x); }

__device__ __inline__ r16 rlog(r16 x) { return hlog(x); }
__device__ __inline__ r32 rlog(r32 x) { return std::log(x); }
__device__ __inline__ r64 rlog(r64 x) { return std::log(x); }

__device__ __inline__ r16 rabs(r16 x) { return __habs(x); }
__device__ __inline__ r32 rabs(r32 x) { return std::abs(x); }
__device__ __inline__ r64 rabs(r64 x) { return std::abs(x); }

template<class T>
__device__ __inline__ T cdiv(T x, T y) { 
  auto u = conjmul(y); return T{ .r = x.r / u, .i = x.i / u };  
}
template<class T>
__device__ __inline__ T cmul(T x, T y) { 
  return T { .r = (x.r * y.r - x.i * y.i), .i = (x.r * y.i + x.i * y.r) };
}

template<class T>
__device__ __inline__ T rsqr(T x) { 
  return x * x;
}

struct MaxOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return (x > y) ? x: y; }
};
struct MinOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return (x < y) ? x: y; }
};
struct AddOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return x + y; }
};
struct MulOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return x * y; }
};
struct DivOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return x / y; }
};
struct SubOP {
  template<class T> static __inline__ __device__ T apply(T x, T y) { return x - y; }
};
struct ClipOP {
  template<class T> static __inline__ __device__ T apply(T x, T lower, T upper) {
    return MinOP::apply(MaxOP::apply(x, lower), upper);
  }
};

template<class T> struct Init;

template<> struct Init<r16> {
  static __device__ __inline__ r16 infinity() { 
    const unsigned short inf_result = 31744;
    return *reinterpret_cast<const r16*>(&inf_result); 
  }
  static __device__ __inline__ r16 epsilon()  { 
    const unsigned short eps_result = 1024;
    return *reinterpret_cast<const r16*>(&eps_result); 
  }
};
template<> struct Init<r32> {
  static constexpr r32 inf_result = std::numeric_limits<r32>::infinity();
  static constexpr r32 eps_result = std::numeric_limits<r32>::epsilon();
  static __device__ __inline__ r32 infinity() { return Init<r32>::inf_result; }
  static __device__ __inline__ r32 epsilon()  { return Init<r32>::eps_result; }
};
template<> struct Init<r64> {
  static constexpr r64 inf_result = std::numeric_limits<r64>::infinity();
  static constexpr r64 eps_result = std::numeric_limits<r64>::epsilon();
  static __device__ __inline__ r64 infinity() { return Init<r64>::inf_result; }
  static __device__ __inline__ r64 epsilon()  { return Init<r64>::eps_result; }
};

// TODO: epsilon value?
template<class T>
__inline__ __device__ bool epsEql(T x, T y) {
  return std::abs(static_cast<double>(x) - static_cast<double>(y)) < 0.001;
}

// TODO:
//  This function has undefined behaviour for threadIdx.x < WARP_SIZE.
//  More research into this __shfl_xor_sync intrinsic needs to be done.
//  In the mean time, if you use this, launch all 32 threads and then
//  for out of bounds reads, parameterize with idempotent value
template <class OP, typename T>
__inline__ __device__ T warpReduce(T value) {
  __syncthreads();

  for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
    value = OP::apply(value, __shfl_xor_sync(0xffffffff, value, mask, WARP_SIZE));
  }
  return value;
}

template <class OP, len_t N, typename T>
__inline__ __device__ T blockReduce(
  T value,
  len_t idx,
  len_t ctrl,
  len_t lim = N
) {
  __shared__ T cache[N];
  
  const T rdx = warpReduce<OP>(value);

  if (ctrl == 0) {
    cache[idx] = rdx;
  }
  __syncthreads();

  if (ctrl == 0 && idx == 0) {
    T tmp = cache[0];
    for (len_t i = 1; i < lim; ++i) { 
      tmp = OP::apply(tmp, cache[i]);
    }
    cache[0] = tmp;
  }
  __syncthreads();
  return cache[0];
}

//template <class OP, len_t N, typename T>
//__inline__ __device__ T blockReduceSerial(
//  T value,
//  len_t idx,
//  len_t ctrl,
//  len_t lim = N
//) {
//  __shared__ T cache[N];
//  
//  const T rdx = warpReduce<OP>(value);
//
//  if (ctrl == 0) {
//    cache[idx] = rdx;
//  }
//  __syncthreads();
//
//  if (ctrl == 0 && idx == 0) {
//    T tmp = cache[0];
//    for (len_t i = 1; i < lim; ++i) { 
//      tmp = OP::apply(tmp, cache[i]);
//    }
//    cache[0] = tmp;
//  }
//  __syncthreads();
//  return cache[0];
//}

#endif // nvcc guard
#endif // c++ guard
#endif // header guard
