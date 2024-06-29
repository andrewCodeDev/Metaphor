#include "../kernel_header.h"

__global__ void __kernel_reduce_ij_i(
    const Scalar* src,
          Scalar* dst,
          Scalar alpha,
    unsigned m,
    unsigned n
){  
    using prc = precision<Scalar>;
    
    // this version covers the m-dimension with blocks
    const unsigned m_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // move to our starting row
    src += n * m_pos;

    dst += (blockDim.y * blockIdx.y);

    Scalar col_sum = 0.0;

    for (unsigned n_step = 0; n_step < n; n_step += blockDim.x) {
        if ((n_step + threadIdx.x < n) && (m_pos < m)) col_sum += src[n_step + threadIdx.x];
    }

    __syncthreads(); // TODO: is this necessary here?

    col_sum = warpReduce<AddOP>(col_sum);

    if ((threadIdx.x == 0) && (m_pos < m)) {
          dst[threadIdx.y] = col_sum + alpha * dst[threadIdx.y];
    }
}

extern "C" void launch_reduce_ij_i_Scalar(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
) {
    // TODO: implement y-block limiting
    const dim3 grid(1, DIMPAD(m, WARP_SIZE));
    const dim3 block(WARP_SIZE, WARP_SIZE);

    __kernel_reduce_ij_i<<<grid, block, 0, get_stream(stream)>>>(
        static_cast<const Scalar*>(src), 
        static_cast<Scalar*>(dst),
        static_cast<Scalar>(alpha),
        static_cast<unsigned>(m),
        static_cast<unsigned>(n)
    );
}
