#include "../kernel_header.h"

__global__ void __kernel_linear_ij_jk_RScalar(
    const RScalar *A,
    const RScalar *B, 
          RScalar alpha, 
    const RScalar *C,
          RScalar beta, 
          RScalar *Y,
    len_t M, 
    len_t N, 
    len_t K 
) {
  __shared__ RScalar tile_A[WARP_SIZE][WARP_SIZE + 1];
  __shared__ RScalar tile_B[WARP_SIZE][WARP_SIZE + 1];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;
  
  // locate our row position on matrix A
  const len_t offset_A = WARP_SIZE * blockIdx.y * N + (N * t_row);
  A += offset_A;
    
  const len_t offset_B = WARP_SIZE * blockIdx.x + (K * t_row);
  B += offset_B;

  const len_t offset_C = (WARP_SIZE * blockIdx.y * K) + (blockIdx.x * WARP_SIZE);
  C += offset_C;
  Y += offset_C;

  // these boundaries don't change, calculate once
  const len_t m_pos = blockIdx.y * WARP_SIZE + t_row;
  const len_t k_pos = blockIdx.x * WARP_SIZE + t_col;

  // this is here for cases of small N and K
  const len_t min_n = min(N, WARP_SIZE);
  const len_t min_k = min(K, WARP_SIZE);

  ////////////////////////////////////////
  // loop and collect resuts in out scalar

  RScalar out = RScalar(0.0f);

  for (len_t step = 0; step < N; step += WARP_SIZE) {
    // check against the row and colum of A
    const len_t n_pos_A = t_col + step;
    const len_t n_pos_B = t_row + step;

    if ((m_pos < M) && (n_pos_A < N))
      tile_A[t_row][t_col] = A[t_col];

    if ((k_pos < K) && (n_pos_B < N))
      tile_B[t_row][t_col] = B[t_col];

    __syncthreads();

    // shift tile over the cols of A
    A += WARP_SIZE;

    // shift tile over the rows of B
    B += WARP_SIZE * K;

    if ((m_pos < M) && (k_pos < K)) {

      // make sure our calculation is in-bounds
      const len_t stop = min(min_n, N - step);
      
      for (len_t i = 0; i < stop; ++i) {
        out += tile_A[t_row][i] * tile_B[i][t_col];
      } 
    }

    __syncthreads();
  }

  C += (K * t_row);
  
  if ((m_pos < M) && (k_pos < K)) {
    Y[t_col] = out * alpha + C[t_col] * beta;
  }
}

extern "C" void launch_matmul_2D_RScalar(
  StreamCtx stream,
  const RScalar *A, 
  const RScalar *B,
        RScalar alpha, // scales product
  const RScalar *C,
        RScalar beta, // blends C back in
        RScalar *Y,
  len_t M, 
  len_t N, 
  len_t K 
) {
    dim3 grid_block(
        DIMPAD(K, WARP_SIZE), 
        DIMPAD(M, WARP_SIZE)
    );

    dim3 tile_block(
        WARP_SIZE,
        WARP_SIZE
    );

    __kernel_linear_ij_jk_RScalar
        <<<grid_block, tile_block, 0, getCtx(stream)>>>(A, B, alpha, C, beta, Y, M, N, K);
}
