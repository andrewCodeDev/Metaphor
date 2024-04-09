#include "../kernel_header.h"

#ifndef __CONV_HELPERS__
#define __CONV_HELPERS__

// TODO: remove these when the the iterative method is replaced

len_t __device__ __host__ __total_passes(len_t k, len_t s) {
    return std::max(1ul, k / s);
}
len_t __device__ __host__ __block_offset(len_t k, len_t s, len_t b) {
    return std::max(s, __total_passes(k, s)) * b;
}
len_t __device__ __host__ __total_blocks(len_t n, len_t k, len_t s) {
    return DIMPAD(n, std::max(s, __total_passes(k, s)));
}

#endif

__global__ void __kernel_convolution_2D_1_channel_source_reverse_RScalar(
          RScalar *src_grads,
    const RScalar *kern_value,
    const RScalar *dst_grads,
    len_t m, // src matrix m dim
    len_t n, // src matrix n dim
    len_t k_dim, // kernel width/height 
    len_t windows, // number of windows over n
    len_t stride // kernel stride
) {

  // TODO: 
  //  Expand this kernel - I ran out of time to make better use
  //  of the actual grid space.   
  
  //  This kernel also works iteratively to prevent data races.
  //  there are several optimization paths that should be explored.

  // This kernel also does not do padding. That needs to get implemented
  // at a later date.

  // starting row position for each thread
  const len_t offset = __block_offset(k_dim, stride, blockIdx.y);

  // total passes each block will make over the n dimension
  const len_t passes = __total_passes(k_dim, stride);
  
  // get this weight one time as it will be the same for each thread
  const RScalar kwgt = (threadIdx.x < k_dim && threadIdx.y < k_dim) 
    ? kern_value[threadIdx.y * k_dim + threadIdx.x] : RScalar(0.0);

  // this boundary is different for each block
  len_t k_m = passes * blockIdx.y;

  const len_t k_stop = k_m + passes;

  for (len_t m_step = offset; (m_step + k_dim) <= m && (k_m < k_stop); m_step += stride, ++k_m) {

    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
      printf("k_m %d, passes %d, offset %d", (int)k_m, (int)passes, (int)offset);

    // This loop tracks how many times we have slid across n
    for (len_t n_step = 0, k_n = 0; (n_step + k_dim) <= n; n_step += stride, ++k_n) {

      if ((threadIdx.x < k_dim) && (threadIdx.y < k_dim)) {

        const RScalar sgrd = dst_grads[k_m * windows + k_n];

        src_grads[n * (m_step + threadIdx.y) + (n_step + threadIdx.x)] += sgrd * kwgt;
      }
    }
    __syncthreads();
  }
}

extern "C" void launch_convolution_2D_source_reverse_RScalar(
  StreamCtx stream,
          RScalar *src_grads,
    const RScalar *kern_value,
    const RScalar *dst_grads,
    len_t m, // src matrix m dim
    len_t n, // src matrix n dim
    len_t k_dim, // kernel width/height 
    len_t windows, // number of windows over n
    len_t stride // kernel stride
) {
    // TODO: 
    //  k_dim must be less than 32 currently
    //  and we need to implement block limiting
    //  only one channel supported currently
  
    dim3 grid(1ul, __total_blocks(m, k_dim, stride));

    dim3 block(WARP_SIZE, WARP_SIZE);

    __kernel_convolution_2D_1_channel_source_reverse_RScalar<<<grid, block, 0, getCtx(stream)>>>(
      src_grads, kern_value, dst_grads, m, n, k_dim, windows, stride
    );
}
