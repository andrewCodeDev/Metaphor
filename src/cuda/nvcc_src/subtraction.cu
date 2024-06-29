#include "../kernel_header.h"

__global__ void __kernel_subtraction(
  const Scalar *dev_a,
  const Scalar *dev_b,
  Scalar *dev_c,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid] = dev_a[tid] - dev_b[tid];
  }
}

extern "C" void launch_subtraction_Scalar(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
) {
  __kernel_subtraction<<<GRID_1D(N), dim3(1024), 0, get_stream(stream)>>>(
    static_cast<const Scalar*>(a), 
    static_cast<const Scalar*>(b), 
    static_cast<Scalar*>(c),
    static_cast<unsigned>(N)
  );
}
