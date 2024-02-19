#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard
__global__ void __kernel_hadamard_r16(
  const r16 *dev_a,
  const r16 *dev_b,
  r16 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] * dev_b[tid];
}

extern "C" void launch_hadamard_r16(
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
) {
  __kernel_hadamard_r16<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_c16(
  const c16 *dev_a,
  const c16 *dev_b,
  c16 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    dev_c[tid].r = (dev_a[tid].r * dev_b[tid].r - dev_a[tid].i * dev_b[tid].i);
    dev_c[tid].i = (dev_a[tid].r * dev_b[tid].i + dev_a[tid].i * dev_b[tid].r);
  }
}

extern "C" void launch_hadamard_c16(
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
) {
  __kernel_hadamard_c16<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard
__global__ void __kernel_hadamard_r32(
  const r32 *dev_a,
  const r32 *dev_b,
  r32 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] * dev_b[tid];
}

extern "C" void launch_hadamard_r32(
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
) {
  __kernel_hadamard_r32<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_c32(
  const c32 *dev_a,
  const c32 *dev_b,
  c32 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    dev_c[tid].r = (dev_a[tid].r * dev_b[tid].r - dev_a[tid].i * dev_b[tid].i);
    dev_c[tid].i = (dev_a[tid].r * dev_b[tid].i + dev_a[tid].i * dev_b[tid].r);
  }
}

extern "C" void launch_hadamard_c32(
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
) {
  __kernel_hadamard_c32<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard
__global__ void __kernel_hadamard_r64(
  const r64 *dev_a,
  const r64 *dev_b,
  r64 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] * dev_b[tid];
}

extern "C" void launch_hadamard_r64(
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
) {
  __kernel_hadamard_r64<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_c64(
  const c64 *dev_a,
  const c64 *dev_b,
  c64 *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    dev_c[tid].r = (dev_a[tid].r * dev_b[tid].r - dev_a[tid].i * dev_b[tid].i);
    dev_c[tid].i = (dev_a[tid].r * dev_b[tid].i + dev_a[tid].i * dev_b[tid].r);
  }
}

extern "C" void launch_hadamard_c64(
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
) {
  __kernel_hadamard_c64<<<GRID_1D(N), 32>>>(a, b, c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

