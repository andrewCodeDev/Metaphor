#include "../kernel_header.h"

// CUDA Kernel for Vector subtraction
__global__ void __kernel_subtraction_r16(
  const r16 *dev_a,
  const r16 *dev_b,
  r16 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] - dev_b[tid];
}

extern "C" void launch_subtraction_r16(
  Stream stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
) {
  __kernel_subtraction_r16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}

__global__ void __kernel_subtraction_c16(
  const c16 *dev_a,
  const c16 *dev_b,
  c16 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid].r = dev_a[tid].r - dev_b[tid].r;
    dev_c[tid].i = dev_a[tid].i - dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_c16(
  Stream stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
) {
  __kernel_subtraction_c16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}
#include "../kernel_header.h"

// CUDA Kernel for Vector subtraction
__global__ void __kernel_subtraction_r32(
  const r32 *dev_a,
  const r32 *dev_b,
  r32 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] - dev_b[tid];
}

extern "C" void launch_subtraction_r32(
  Stream stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
) {
  __kernel_subtraction_r32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}

__global__ void __kernel_subtraction_c32(
  const c32 *dev_a,
  const c32 *dev_b,
  c32 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid].r = dev_a[tid].r - dev_b[tid].r;
    dev_c[tid].i = dev_a[tid].i - dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_c32(
  Stream stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
) {
  __kernel_subtraction_c32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}
#include "../kernel_header.h"

// CUDA Kernel for Vector subtraction
__global__ void __kernel_subtraction_r64(
  const r64 *dev_a,
  const r64 *dev_b,
  r64 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] - dev_b[tid];
}

extern "C" void launch_subtraction_r64(
  Stream stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
) {
  __kernel_subtraction_r64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}

__global__ void __kernel_subtraction_c64(
  const c64 *dev_a,
  const c64 *dev_b,
  c64 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid].r = dev_a[tid].r - dev_b[tid].r;
    dev_c[tid].i = dev_a[tid].i - dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_c64(
  Stream stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
) {
  __kernel_subtraction_c64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, c, N);
}
