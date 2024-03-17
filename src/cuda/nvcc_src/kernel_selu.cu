#include "../kernel_header.h"

__global__ void __kernel_selu_RScalar(
  const RScalar *a,
        RScalar *b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const RScalar value = a[tid];    
    b[tid] = (value >= RScalar(0.0f)) ? value : rexp(value) - RScalar(1.0f);
  }
}

extern "C" void launch_selu_RScalar(
  StreamCtx stream,
  const RScalar* a,
        RScalar* b, 
  len_t N
) {
  __kernel_selu_RScalar<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(a, b, N);
}
