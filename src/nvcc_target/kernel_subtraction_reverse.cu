#include "../kernel_header.h"

__global__ void __kernel_subtraction_reverse_r16(
  r16 *dev_a,
  const r16 *dev_b,
  r16 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += coef * dev_b[tid];
}

extern "C" void launch_subtraction_reverse_r16(
  r16* a, 
  const r16* b, 
  const r16 coef,
  len_t N
) {
  __kernel_subtraction_reverse_r16<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_subtraction_reverse_c16(
  c16 *dev_a,
  const c16 *dev_b,
  const r16 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += coef * dev_b[tid].r;
    dev_a[tid].i += coef * dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_reverse_c16(
  c16* a, 
  const c16* b, 
  const r16 coef,
  len_t N
) {
  __kernel_subtraction_reverse_c16<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

__global__ void __kernel_subtraction_reverse_r32(
  r32 *dev_a,
  const r32 *dev_b,
  r32 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += coef * dev_b[tid];
}

extern "C" void launch_subtraction_reverse_r32(
  r32* a, 
  const r32* b, 
  const r32 coef,
  len_t N
) {
  __kernel_subtraction_reverse_r32<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_subtraction_reverse_c32(
  c32 *dev_a,
  const c32 *dev_b,
  const r32 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += coef * dev_b[tid].r;
    dev_a[tid].i += coef * dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_reverse_c32(
  c32* a, 
  const c32* b, 
  const r32 coef,
  len_t N
) {
  __kernel_subtraction_reverse_c32<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

__global__ void __kernel_subtraction_reverse_r64(
  r64 *dev_a,
  const r64 *dev_b,
  r64 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += coef * dev_b[tid];
}

extern "C" void launch_subtraction_reverse_r64(
  r64* a, 
  const r64* b, 
  const r64 coef,
  len_t N
) {
  __kernel_subtraction_reverse_r64<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_subtraction_reverse_c64(
  c64 *dev_a,
  const c64 *dev_b,
  const r64 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += coef * dev_b[tid].r;
    dev_a[tid].i += coef * dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_reverse_c64(
  c64* a, 
  const c64* b, 
  const r64 coef,
  len_t N
) {
  __kernel_subtraction_reverse_c64<<<GRID_1D(N), 32>>>(a, b, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

