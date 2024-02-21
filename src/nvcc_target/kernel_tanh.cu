#include "../kernel_header.h"

__global__ void __kernel_tanh_r16(
  const r16 *a_value,
        r16 *b_value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_tanh_r16(
  Stream stream,
  const r16* a,
        r16* b, 
  len_t N
) {
  __kernel_tanh_r16<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
#include "../kernel_header.h"

__global__ void __kernel_tanh_r32(
  const r32 *a_value,
        r32 *b_value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_tanh_r32(
  Stream stream,
  const r32* a,
        r32* b, 
  len_t N
) {
  __kernel_tanh_r32<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
#include "../kernel_header.h"

__global__ void __kernel_tanh_r64(
  const r64 *a_value,
        r64 *b_value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_tanh_r64(
  Stream stream,
  const r64* a,
        r64* b, 
  len_t N
) {
  __kernel_tanh_r64<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
