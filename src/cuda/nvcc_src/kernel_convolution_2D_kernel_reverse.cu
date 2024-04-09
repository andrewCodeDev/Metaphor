#include "../kernel_header.h"

__global__ void __kernel_convolution_2D_1_channel_reverse_kernel_RScalar(
    const RScalar *src_value,
          RScalar *kern_grads,
    const RScalar *dst_grads,
          RScalar *scratch,
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
  auto grid = cg::this_grid();

  __shared__ RScalar smem[WARP_SIZE][WARP_SIZE + 1];

  smem[threadIdx.y][threadIdx.x] = RScalar(0.0);
  
  for (len_t m_step = 0, d_step = 0; (m_step + k_dim) <= m; m_step += stride, ++d_step) {

    const auto src_tmp = src_value + n * (m_step + threadIdx.y);

    //const auto dst_tmp = dst_grads + n * (m_step + threadIdx.y);

    // This loop tracks how many times we have slid across n
    for (len_t w_step = 0; (w_step + blockIdx.x) < windows; w_step += gridDim.x) {

      if (threadIdx.x < k_dim && threadIdx.y < k_dim) {

        const len_t n_pos = stride * (blockIdx.x + w_step);
        const auto dst_tmp = dst_grads + d_step * windows + w_step;

        const RScalar src_val = src_tmp[n_pos];
        const RScalar dst_grd = dst_tmp[w_step + blockIdx.x];

        smem[threadIdx.y][threadIdx.x] += src_val * dst_grd;
      }
    }
  }

  // scratch is (gridDim.x * k_dim) wide and k_dim tall...
  // any extra blocks have been reduce into shared memory
  const len_t s_col = gridDim.x * k_dim; 

  if (threadIdx.x < k_dim && threadIdx.y < k_dim) {
    // each block has a dedicated segment in shared memory for its collected values
    scratch[(s_col * threadIdx.y) + (blockIdx.x * k_dim) + threadIdx.x] = smem[threadIdx.y][threadIdx.x];
  }

  grid.sync();  

  const len_t lower = blockIdx.x * 2;
  const len_t upper = lower + 1;

  for (unsigned limit = gridDim.x; limit > 1; limit = (limit + 1) / 2) {

      grid.sync();

      // prevents data race for columns reading and writing
      RScalar __attribute__((unused)) lower_value = RScalar(0.0f); 
      RScalar __attribute__((unused)) upper_value = RScalar(0.0f); 

      if ((lower < limit) && (threadIdx.x < k_dim) && (threadIdx.y < k_dim)) {                    
          lower_value = scratch[(s_col * threadIdx.y) + (lower * k_dim) + threadIdx.x];
      }
      if ((upper < limit) && (threadIdx.x < k_dim) && (threadIdx.y < k_dim)) {
          upper_value = scratch[(s_col * threadIdx.y) + (upper * k_dim) + threadIdx.x];
      }

      grid.sync();

      if ((lower < limit) && (threadIdx.x < k_dim) && (threadIdx.y < k_dim)) {                    
          scratch[(s_col * threadIdx.y) + (blockIdx.x * k_dim) + threadIdx.x] = lower_value + upper_value;
      } 
  }

  if ((blockIdx.x == 0) && (threadIdx.x < k_dim) && (threadIdx.y < k_dim)) {
    kern_grads[threadIdx.y * k_dim + threadIdx.x] = scratch[(s_col * threadIdx.y) + threadIdx.x];
  }
}

extern "C" void launch_convolution_2D_reverse_kernel_RScalar(
  StreamCtx stream,
    const RScalar *src_value,
          RScalar *kern_grads,
    const RScalar *dst_grads,
          RScalar *scratch,
    len_t m, // src matrix m dim
    len_t n, // src matrix n dim
    len_t k_dim, // kernel width/height 
    len_t windows,
    len_t stride // kernel stride
) {
    // TODO: 
    //  k_dim must be less than 32 currently
    //  only 1 channel supported currently

    dim3 grid(std::min(windows, 1024ul), 1ul);

    dim3 block(WARP_SIZE, WARP_SIZE);

    void* args[] = { 
      (void*)&src_value,
      (void*)&kern_grads,
      (void*)&dst_grads,
      (void*)&scratch,
      (void*)&m,
      (void*)&n,
      (void*)&k_dim,
      (void*)&windows,
      (void*)&stride
    };

    CUDA_ASSERT(cudaLaunchCooperativeKernel(
      (void*)(__kernel_convolution_2D_1_channel_reverse_kernel_RScalar), grid, block, args, 0, getCtx(stream)
    )); 
}
