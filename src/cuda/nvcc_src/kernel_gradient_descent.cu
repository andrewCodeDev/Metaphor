#include "../kernel_header.h"

__global__ void __kernel_gradient_descent_RScalar(
        RScalar *a_value,
  const RScalar *a_grads,
  RScalar rate,
  RScalar lower,
  RScalar upper,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
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
  len_t N
) {
  __kernel_gradient_descent_RScalar<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(
    a_value, a_grads, rate, lower, upper, N
  );
}
