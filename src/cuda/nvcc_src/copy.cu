#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_copy_Scalar(
  const Scalar* src,
        Scalar* dst,
  unsigned n
) {
  // TODO: Implement block limiting
  
  // each thread loads 4 elements in each block
  const unsigned chunk_size = (blockDim.x * 4);

  // find our starting position
  const unsigned chunk_offset = chunk_size * blockIdx.x;

  // find thread position within chunk
  const unsigned pos = chunk_offset + (threadIdx.x * 4);

  if (pos < n) {
    // check if we have enough room for a coalesced load
    if (4 < (n - pos)) {
      auto src_cls = *reinterpret_cast<coalesce<Scalar>::c_ptr>(&src[pos]);
      auto dst_cls =  reinterpret_cast<coalesce<Scalar>::ptr>(&dst[pos]);
      *dst_cls = src_cls;
    }
    else {
      for (unsigned i = pos; i < n; ++i) dst[i] = src[i];
    }
  }
}

extern "C" void launch_copy_Scalar(
  const void* src, 
        void* dst, 
  len_t n,
  StreamContext stream
) {
  // TODO: search for hyper parameters
  dim3 grid_block(DIMPAD(n, (1024 * 4)), 1);
  dim3 thread_block(1024);
  __kernel_copy_Scalar<<<grid_block, thread_block, 0, get_stream(stream)>>>(
    static_cast<const Scalar*>(src), 
    static_cast<Scalar*>(dst), 
    static_cast<unsigned>(n)
  );
}


