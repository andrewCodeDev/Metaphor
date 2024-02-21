#include "../kernel_header.h"

__global__ void __kernel_addition_reverse_r16(
  r16 *dev_a,
  const r16 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += dev_b[tid];
}

extern "C" void launch_addition_reverse_r16(
  Stream stream,
  r16* a, 
  const r16* b, 
  len_t N
) {
  __kernel_addition_reverse_r16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}

__global__ void __kernel_addition_reverse_c16(
  c16 *dev_a,
  const c16 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += dev_b[tid].r;
    dev_a[tid].i += dev_b[tid].i;
  }
}

extern "C" void launch_addition_reverse_c16(
  Stream stream,
  c16* a, 
  const c16* b, 
  len_t N
) {
  __kernel_addition_reverse_c16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
#include "../kernel_header.h"

__global__ void __kernel_addition_reverse_r32(
  r32 *dev_a,
  const r32 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += dev_b[tid];
}

extern "C" void launch_addition_reverse_r32(
  Stream stream,
  r32* a, 
  const r32* b, 
  len_t N
) {
  __kernel_addition_reverse_r32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}

__global__ void __kernel_addition_reverse_c32(
  c32 *dev_a,
  const c32 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += dev_b[tid].r;
    dev_a[tid].i += dev_b[tid].i;
  }
}

extern "C" void launch_addition_reverse_c32(
  Stream stream,
  c32* a, 
  const c32* b, 
  len_t N
) {
  __kernel_addition_reverse_c32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
#include "../kernel_header.h"

__global__ void __kernel_addition_reverse_r64(
  r64 *dev_a,
  const r64 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += dev_b[tid];
}

extern "C" void launch_addition_reverse_r64(
  Stream stream,
  r64* a, 
  const r64* b, 
  len_t N
) {
  __kernel_addition_reverse_r64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}

__global__ void __kernel_addition_reverse_c64(
  c64 *dev_a,
  const c64 *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += dev_b[tid].r;
    dev_a[tid].i += dev_b[tid].i;
  }
}

extern "C" void launch_addition_reverse_c64(
  Stream stream,
  c64* a, 
  const c64* b, 
  len_t N
) {
  __kernel_addition_reverse_c64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
