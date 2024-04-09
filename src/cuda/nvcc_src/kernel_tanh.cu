#include "../kernel_header.h"

__global__ void __kernel_tanh_RScalar(
  const RScalar *a_value,
        RScalar *b_value,
  len_t n
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_tanh_RScalar(
  StreamCtx stream,
  const RScalar* a,
        RScalar* b, 
  len_t n
) {
  __kernel_tanh_RScalar<<<DIMPAD(n, 1024), dim3(1024), 0, getCtx(stream)>>>(a, b, n);
}
