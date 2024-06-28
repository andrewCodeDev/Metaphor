#ifndef __COMPLEX_TYPES_H__
#define __COMPLEX_TYPES_H__

#define MAX_DIMS 6
typedef unsigned long len_t;
const len_t WARP_SIZE = 32;
const len_t UINT_BUFFER_SIZE = 32;

#include <stdint.h>

// coalesced types for load optimization

// fundamental types

// use the cuda definitions with device
// defined member functions.
#if defined(__cplusplus)
  #include "cuda_fp16.h"

  typedef __half r16;
#else
// only the size really matters here
// when we compile cuda, we do it via
// nvcc which defines __cplusplus. The
// host code will never use this for
// floating point operations.
typedef struct {
    unsigned short __x;
} r16;
#endif

typedef float  r32;
typedef double r64;

// C++ helper classes
#if defined(__cplusplus)

template <class T>
struct coalesce {
  using c_ptr = const coalesce<T>*;
  using ptr = coalesce<T>*;
  static constexpr len_t warp_step = WARP_SIZE / 4;
  T w, x, y, z;

  __device__ coalesce& operator+=(coalesce const& c) {
    w += c.w; x += c.x; y += c.y; z += c.z; return *this;
  }
};

// used to adjust precision up on r16
template <class T> struct precision;

template <> struct precision<r16>{ 
  using type = r32; 
  __device__ __inline__ static constexpr type one() { return 1.0; };
  __device__ __inline__ static constexpr type zero() { return 0.0; };
  static __device__ __inline__ type cast(type x){ return x; }};

template <> struct precision<r32>{ 
  using type = r32; 
  __device__ __inline__ static constexpr type one() { return 1.0; };
  __device__ __inline__ static constexpr type zero() { return 0.0; };
  static __device__ __inline__ type cast(type x){ return x; }};

template <> struct precision<r64>{ 
  using type = r64; 
  __device__ __inline__ static constexpr type one() { return 1.0; };
  __device__ __inline__ static constexpr type zero() { return 0.0; };
  static __device__ __inline__ type cast(type x){ return x; }};

#else
#endif


typedef struct {
  r16 r;
  r16 i;
} c16;

typedef struct {
  r32 r;
  r32 i;
} c32;

typedef struct {
  r64 r;
  r64 i;
} c64;

/////////////////////////
// used for key sorting

// TODO:
//  Consider making these only r32
//  The padding doesn't make sense
//  for larger types. Could cause
//  problems for large r64?

typedef struct {
  r16 val;
  unsigned key;
} SortPair_r16;


typedef struct {
  r32 val;
  unsigned key;
} SortPair_r32;


typedef struct {
  r64 val;
  unsigned key;
} SortPair_r64;

#endif
