#ifndef __COMPLEX_TYPES_H__
#define __COMPLEX_TYPES_H__

#define MAX_DIMS 6
typedef unsigned long len_t;
const len_t WARP_SIZE = 32;
const len_t UINT_BUFFER_SIZE = 32;

// coalesced types for load optimization

// fundamental types

// use the cuda definitions with device
// defined member functions.
#if defined(__cplusplus)
  #include "cuda/cuda_fp16.h"

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
};

// used to adjust precision up on r16
template <class T> struct precision;
template <> struct precision<r16>{ using type = r32; static __device__ type cast(type x){ return x; }};
template <> struct precision<r32>{ using type = r32; static __device__ type cast(type x){ return x; }};
template <> struct precision<r64>{ using type = r64; static __device__ type cast(type x){ return x; }};

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

////////////////////////
// C tensors variants

typedef struct {
  r16* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} RTensor16;

typedef struct {
  r32* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} RTensor32;

typedef struct {
  r64* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} RTensor64;

typedef struct {
  c16* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} CTensor16;

typedef struct {
  c32* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} CTensor32;

typedef struct {
  c64* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} CTensor64;

typedef struct {
  char* values;
  len_t sizes[MAX_DIMS];
  len_t strides[MAX_DIMS];
  len_t dims;
  len_t len;
} QTensor8;

typedef struct {
  len_t order[MAX_DIMS];
} Permutation;

// for small buffer optimizations
typedef struct {
  unsigned items[UINT_BUFFER_SIZE];
  unsigned used;
} UintBuffer;

#endif
