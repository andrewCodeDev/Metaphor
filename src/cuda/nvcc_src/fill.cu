#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_fill(
  Scalar *dev_a,
  Scalar value,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    dev_a[tid] = value;
  }
}

extern "C" void launch_fill_Scalar(
  void* dev_a,
  double value, 
  len_t N,
  StreamContext stream
) {
  __kernel_fill<<<GRID_1D(N), dim3(1024), 0, get_stream(stream)>>>(
    static_cast<Scalar*>(dev_a), 
    static_cast<Scalar>(value), 
    static_cast<unsigned>(N)
  );
}
