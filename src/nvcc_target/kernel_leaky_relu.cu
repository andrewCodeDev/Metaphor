#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_r16(
  const r16 *a_value,
        r16 *b_value,
        r16 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // prevents ambiguous < lookup
  const r16 zero = 0.0;
     
  if (tid < N) {
    const r16 u = a_value[tid];
    b_value[tid] = (u < zero) ? u * coef : u;
  }
}

extern "C" void launch_leaky_relu_r16(
  Stream stream,
  const r16* a,
        r16* b, 
        r16 coef,
  len_t N
) {
  __kernel_leaky_relu_r16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, coef, N);
}
#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_r32(
  const r32 *a_value,
        r32 *b_value,
        r32 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // prevents ambiguous < lookup
  const r32 zero = 0.0;
     
  if (tid < N) {
    const r32 u = a_value[tid];
    b_value[tid] = (u < zero) ? u * coef : u;
  }
}

extern "C" void launch_leaky_relu_r32(
  Stream stream,
  const r32* a,
        r32* b, 
        r32 coef,
  len_t N
) {
  __kernel_leaky_relu_r32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, coef, N);
}
#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_r64(
  const r64 *a_value,
        r64 *b_value,
        r64 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // prevents ambiguous < lookup
  const r64 zero = 0.0;
     
  if (tid < N) {
    const r64 u = a_value[tid];
    b_value[tid] = (u < zero) ? u * coef : u;
  }
}

extern "C" void launch_leaky_relu_r64(
  Stream stream,
  const r64* a,
        r64* b, 
        r64 coef,
  len_t N
) {
  __kernel_leaky_relu_r64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, coef, N);
}
