#include "../kernel_header.h"

__global__ void __kernel_convolution_2D_1_channel_RScalar(
    const RScalar *src,
    const RScalar *kern,
          RScalar *dst,
    len_t m, // src matrix m dim
    len_t n, // src matrix n dim
    len_t k_dim, // kernel width/height 
    len_t windows, // number of windows over n
    len_t stride // kernel stride
) {

  // TODO: 
  //  expand this kernel - I ran out of time to make better use
  //  of the actual grid space. We're just walking a single column
  //  of kernels across the rows of the src matrix.

  // This kernel also does not do padding. That needs to get implemented
  // at a later date.
  
  // get this weight one time as it will be the same for each thread
  const RScalar kwgt = (threadIdx.x < k_dim && threadIdx.y < k_dim) 
    ? kern[threadIdx.y * k_dim + threadIdx.x] : RScalar(0.0);

  for (len_t m_step = 0, d_step = 0; (m_step + k_dim) <= m; m_step += stride, ++d_step) {

    const auto src_tmp = src + n * (m_step + threadIdx.y);

    // This loop tracks how many times we have slid across n
    for (len_t w_step = 0; (w_step + blockIdx.x) < windows; w_step += gridDim.x) {

      // windows tells us the n dimension of the output
      const auto dst_tmp = dst + d_step * windows + w_step;

      const len_t n_pos = stride * (blockIdx.x + w_step);

      const RScalar src_val = (threadIdx.x < k_dim) 
        ? src_tmp[n_pos] : RScalar(0.0);

      const RScalar conv = blockReduce<AddOP, WARP_SIZE>(
        src_val * kwgt, threadIdx.x, threadIdx.y, k_dim
      );

      if ((threadIdx.x == 0 && threadIdx.y == 0)) {
        dst_tmp[w_step + blockIdx.x] = conv;
      }
    }
  }
}

extern "C" void launch_convolution_2D_RScalar(
  StreamCtx stream,
    const RScalar *src,
    const RScalar *kern,
          RScalar *dst,
    len_t m, // src matrix m dim
    len_t n, // src matrix n dim
    len_t k_dim, // kernel width/height 
    len_t windows,
    len_t stride // kernel stride
) {
    // TODO: 
    //  k_dim must be less than 32 currently
    //  only 1 channel reverse supported currently

    dim3 grid_block(std::min(windows, 1024ul), 1ul);

    dim3 tile_block(WARP_SIZE, WARP_SIZE);

    __kernel_convolution_2D_1_channel_RScalar<<<grid_block, tile_block, 0, getCtx(stream)>>>(
      src, kern, dst, m, n, k_dim, windows, stride
    );
}
