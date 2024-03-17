#include "../kernel_header.h"

__global__ void __kernel_outer_product_i_j_RScalar(
  const RScalar* x,
  const RScalar* y,
  RScalar alpha,
        RScalar* A,
  RScalar beta,
  len_t M,
  len_t N
) {
  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;
  
  // this boundary doesn't change, calculate once
  const len_t m_pos = blockIdx.y * WARP_SIZE + t_row;
  const len_t n_pos = blockIdx.x * WARP_SIZE + t_col;

  // shift x over the row space
  x += (blockIdx.y * WARP_SIZE);

  // shift y over the column space
  y += (blockIdx.x * WARP_SIZE);

  // shift A over the row space + current row
  A += N * (blockIdx.y * WARP_SIZE + t_row);

  // shift A over the column space
  A += (blockIdx.x * WARP_SIZE);

  if ((m_pos < M) && (n_pos < N)) {
     A[t_col] = x[t_row] * y[t_col] * alpha + A[t_col] * beta;
  }
}

extern "C" void launch_outer_product_i_j_RScalar(
  StreamCtx stream,
  const RScalar *x,
  const RScalar *y, 
        RScalar alpha, // scales product
        RScalar *A,
        RScalar beta, // blends A back in
  len_t M, 
  len_t N
) {
    dim3 grid_block(
        DIMPAD(N, WARP_SIZE), 
        DIMPAD(M, WARP_SIZE)
    );

    dim3 tile_block(
        WARP_SIZE,
        WARP_SIZE
    );

    __kernel_outer_product_i_j_RScalar
        <<<grid_block, tile_block, 0, getCtx(stream)>>>(x, y, alpha, A, beta, M, N);
}
