#include "../kernel_header.h"

__global__ void __kernel_reduce_ij_i_RScalar(
    const RScalar* src,
          RScalar* dst,
          RScalar alpha,
    len_t m,
    len_t n
){  
    // this version covers the m-dimension with blocks
    const unsigned m_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // move to our starting row
    src += n * (threadIdx.y + (blockDim.y * blockIdx.y));
    dst += (blockDim.y * blockIdx.y);

    RScalar col_sum = 0.0;

    for (unsigned n_step = 0; n_step < n; n_step += blockDim.x) {
      if ((n_step + threadIdx.x < n) && (m_pos < m)) col_sum += src[n_step + threadIdx.x];
    }

    __syncthreads(); // TODO: is this necessary here?

    col_sum += warpReduce<AddOP>(col_sum);

    if (threadIdx.x == 0 && (m_pos < m)) {
      dst[threadIdx.y] = col_sum + alpha * dst[threadIdx.y];
    }
}

extern "C" void launch_reduce_ij_i_RScalar(
    StreamCtx stream,
    const RScalar* src,
          RScalar* dst,
          RScalar alpha,
    len_t m,
    len_t n
) {
    // TODO: implement y-block limiting
    const dim3 grid(1, DIMPAD(m, WARP_SIZE));
    const dim3 block(WARP_SIZE, WARP_SIZE);
    __kernel_reduce_ij_i_RScalar<<<grid, block, 0, getCtx(stream)>>>(src, dst, alpha, m, n);
}
