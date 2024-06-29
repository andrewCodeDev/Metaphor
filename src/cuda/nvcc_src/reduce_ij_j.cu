#include "../kernel_header.h"

__global__ void __kernel_reduce_ij_j(
    const Scalar* src,
          Scalar* dst,
          Scalar alpha,
    len_t m,
    len_t n
){
    __shared__ Scalar smem[WARP_SIZE][WARP_SIZE + 1];
  
    // this version covers the n-dimension with blocks
    const unsigned n_pos = blockIdx.x * blockDim.x + threadIdx.x;

    // move to our starting row and column
    src += threadIdx.y * n + (blockDim.x * blockIdx.x);
    dst += (blockDim.x * blockIdx.x);

    Scalar col_sum = 0.0;

    for (unsigned m_step = 0; m_step < m; m_step += blockDim.y) {

      // TODO: create transpose boundary conditions to reduce smem reads
      smem[threadIdx.y][threadIdx.x] = 0.0f;

      const unsigned m_pos = m_step + threadIdx.y;

      if ((m_pos < m) && (n_pos < n)) {
        // transpose our load in
        smem[threadIdx.y][threadIdx.x] = src[threadIdx.x];
      }

      src += (blockDim.y * n);

      __syncthreads();

      col_sum += warpReduce<AddOP>(smem[threadIdx.x][threadIdx.y]);

      __syncthreads();
    }

    if (threadIdx.x == 0 && (threadIdx.y + blockDim.x * blockIdx.x) < n) {
      dst[threadIdx.y] = col_sum + alpha * dst[threadIdx.x];
    }

}

extern "C" void launch_reduce_ij_j_Scalar(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
) {
    const dim3 grid(DIMPAD(m, WARP_SIZE));
    const dim3 block(WARP_SIZE, WARP_SIZE);

    __kernel_reduce_ij_j<<<grid, block, 0, get_stream(stream)>>>(
        static_cast<const Scalar*>(src), 
        static_cast<Scalar*>(dst),
        static_cast<Scalar>(alpha),
        static_cast<unsigned>(m),
        static_cast<unsigned>(n)
    );
}
