
#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

__global__ void __kernel_broadcast_i_ij(
  const Scalar* __restrict src, // input
        Scalar* __restrict dst, // output
        Scalar alpha,
  unsigned m,
  unsigned n
) {
  using prc = precision<Scalar>;

  Scalar __attribute__((unused)) tmp;

  // this loop moves our blocks down the rows
  for (len_t m_step = 0; m_step < m; m_step += (gridDim.y * blockDim.y))
  {
    // check if we're still in bounds for the rows
    const len_t m_pos = threadIdx.y + (blockIdx.y * blockDim.y) + m_step;

    if (m_pos < m)
      tmp = src[m_pos];

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        dst[m_pos * n + n_pos] = tmp + alpha * dst[m_pos * n + n_pos];
      }
    }
  }
}

extern "C" void launch_broadcast_i_ij_Scalar(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
) {
    const dim3 grid(DIMPAD(m, WARP_SIZE));
    const dim3 block(WARP_SIZE, WARP_SIZE);

    __kernel_broadcast_i_ij<<<grid, block, 0, get_stream(stream)>>>(
        static_cast<const Scalar*>(src), 
        static_cast<Scalar*>(dst),
        static_cast<Scalar>(alpha),
        static_cast<unsigned>(m),
        static_cast<unsigned>(n)
    );
}
