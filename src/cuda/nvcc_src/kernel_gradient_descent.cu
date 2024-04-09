#include "../kernel_header.h"

__global__ void __kernel_gradient_descent_RScalar(
        RScalar *a_value,
  const RScalar *a_grads,
  RScalar rate,
  RScalar lower,
  RScalar upper,
  len_t n
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < n) {
    a_value[tid] -= ClipOP::apply(a_grads[tid], lower, upper) * rate;
  }
}

extern "C" void launch_gradient_descent_RScalar(
  StreamCtx stream,
        RScalar* a_value,
  const RScalar* a_grads, 
  RScalar rate,
  RScalar lower,
  RScalar upper,
  len_t n
) {
  __kernel_gradient_descent_RScalar<<<DIMPAD(n, 1024), dim3(1024), 0, getCtx(stream)>>>(
    a_value, a_grads, rate, lower, upper, n
  );
}
