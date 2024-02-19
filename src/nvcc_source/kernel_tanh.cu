#include "../kernel_header.h"

__global__ void __kernel_tanh_RScalar(
  const RScalar *a_value,
        RScalar *b_value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_tanh_RScalar(
  const RScalar* a,
        RScalar* b, 
  len_t N
) {
  __kernel_tanh_RScalar<<<GRID_1D(N), 32>>>(a, b, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}
