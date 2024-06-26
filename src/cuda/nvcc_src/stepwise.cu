#include "../kernel_header.h"

__global__ void __kernel_stepwise(
  const Scalar *a,
        Scalar *b,
  unsigned n
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n) {
    const Scalar value = a[tid];    
    b[tid] = (value >= Scalar(0.0)) ? value : Scalar(0.0);
  }
}

extern "C" void launch_stepwise_Scalar(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
) {
  __kernel_stepwise<<<DIMPAD(n, 1024), dim3(1024), 0, get_stream(stream)>>>(
    static_cast<const Scalar*>(a),
    static_cast<Scalar*>(b),
    static_cast<unsigned>(n)
  );
}
