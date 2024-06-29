#include "../kernel_header.h"

__global__ void __kernel_negate(
  const Scalar *dev_a,
        Scalar *dev_b,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_b[tid] = -dev_a[tid];
  }
}

extern "C" void launch_negate_Scalar(
  const void* a,
        void* b, 
  len_t N,
  StreamContext stream
) {
  __kernel_negate<<<GRID_1D(N), dim3(1024), 0, get_stream(stream)>>>(
    static_cast<const Scalar*>(a), 
    static_cast<Scalar*>(b),
    static_cast<unsigned>(N)
  );
}
