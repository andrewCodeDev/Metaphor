#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_fill_r16(
  r16 *dev_a,
  r16 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] = value;
}

extern "C" void launch_fill_r16(
  Stream stream,
  r16* dev_a,
  r16 value, 
  len_t N
) {
  __kernel_fill_r16<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}

__global__ void __kernel_fill_c16(
  c16 *dev_a,
  c16 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r = value.r;
    dev_a[tid].i = value.i;
  }
}

extern "C" void launch_fill_c16(
  Stream stream,
  c16* dev_a,
  c16 value, 
  len_t N
) {
  __kernel_fill_c16<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}


#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_fill_r32(
  r32 *dev_a,
  r32 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] = value;
}

extern "C" void launch_fill_r32(
  Stream stream,
  r32* dev_a,
  r32 value, 
  len_t N
) {
  __kernel_fill_r32<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}

__global__ void __kernel_fill_c32(
  c32 *dev_a,
  c32 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r = value.r;
    dev_a[tid].i = value.i;
  }
}

extern "C" void launch_fill_c32(
  Stream stream,
  c32* dev_a,
  c32 value, 
  len_t N
) {
  __kernel_fill_c32<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}


#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_fill_r64(
  r64 *dev_a,
  r64 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] = value;
}

extern "C" void launch_fill_r64(
  Stream stream,
  r64* dev_a,
  r64 value, 
  len_t N
) {
  __kernel_fill_r64<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}

__global__ void __kernel_fill_c64(
  c64 *dev_a,
  c64 value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r = value.r;
    dev_a[tid].i = value.i;
  }
}

extern "C" void launch_fill_c64(
  Stream stream,
  c64* dev_a,
  c64 value, 
  len_t N
) {
  __kernel_fill_c64<<<1, GRID_1D(N), 32, getStream(stream)>>>(dev_a, value, N);
}


