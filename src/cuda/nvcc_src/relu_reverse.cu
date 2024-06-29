#include "../kernel_header.h"

__global__ void __kernel_relu_reverse(
  const Scalar *a_value,
        Scalar *a_grads,
  const Scalar *b_grads,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const Scalar a = a_value[tid];    
    const Scalar g = b_grads[tid];
    a_grads[tid] += g * ((a >= Scalar(0.0)) ? Scalar(1.0) : Scalar(0.0));
  }
}

extern "C" void launch_relu_reverse_Scalar(
  const void *a_value,
        void *a_grads,
  const void *b_grads,
  len_t N,
  StreamContext stream
) {
  __kernel_relu_reverse<<<GRID_1D(N), dim3(1024), 0, get_stream(stream)>>>(
    static_cast<const Scalar*>(a_value),
          static_cast<Scalar*>(a_grads),
    static_cast<const Scalar*>(b_grads), 
    static_cast<unsigned>(N)
  );
}
