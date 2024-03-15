#include "../kernel_header.h"

__global__ void __kernel_selu_reverse_RScalar(
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const RScalar b = b_value[tid];    
    const RScalar g = b_grads[tid];
    // for (x < 0) -> e^x - 1, dx is e^x. To recover e^x, add 1 to x
    a_grads[tid] += (b >= RScalar(0.0f)) ? RScalar(1.0f) : b + RScalar(1.0f);
  }
}

extern "C" void launch_selu_reverse_RScalar(
  StreamCtx stream,
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  __kernel_selu_reverse_RScalar<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(
    a_grads, b_value, b_grads, N
  );
}
