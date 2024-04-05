#include "../kernel_header.h"

// right hand transposed linear transform
__global__ void __kernel_linear_ij_kj_RScalar(
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
  
  // locate our row position on matrix A and B
  A += WARP_SIZE * blockIdx.y * N + (N * t_row);
  B += WARP_SIZE * blockIdx.x * N + (N * t_row);
  
  C += (WARP_SIZE * blockIdx.y * K) + (blockIdx.x * WARP_SIZE);
  Y += (WARP_SIZE * blockIdx.y * K) + (blockIdx.x * WARP_SIZE);

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
    const len_t n_pos = t_col + step;

    if ((m_pos < M) && (n_pos < N))
      tile_A[t_row][t_col] = A[t_col];

    // m_pos just measure the left side dimension
    // since B is transpose, this measures the k
    // boundary for A as well. K_pos deals with
    // the output on C

    if ((m_pos < K) && (n_pos < N))
      tile_B[t_row][t_col] = B[t_col];

    __syncthreads();

    // shift tile over the cols of A
    A += WARP_SIZE;

    // shift tile over the cols of B
    B += WARP_SIZE;

    if ((m_pos < M) && (k_pos < K)) {

      // make sure our calculation is in-bounds
      const len_t stop = min(min_n, N - step);
      
      // TODO: This has many bank conflicts
      for (len_t i = 0; i < stop; ++i) {
        out += tile_A[t_row][i] * tile_B[t_col][i];
      } 
    }

    __syncthreads();
  }

  C += (K * t_row);
  Y += (K * t_row);
  
  if ((m_pos < M) && (k_pos < K)) {
    Y[t_col] = out * alpha + C[t_col] * beta;
  }
}

extern "C" void launch_linear_ij_kj_RScalar(
  StreamCtx stream,
  const RScalar *A, 
  const RScalar *B,
        RScalar alpha, // scales product
  const RScalar *C,
        RScalar beta, // blends C back in
        RScalar *Y,
  len_t m, 
  len_t n, 
  len_t k 
) {
    dim3 grid_block(
        DIMPAD(k, WARP_SIZE), 
        DIMPAD(m, WARP_SIZE)
    );

    dim3 tile_block(
        WARP_SIZE,
        WARP_SIZE
    );

    __kernel_linear_ij_kj_RScalar
        <<<grid_block, tile_block, 0, getCtx(stream)>>>(A, B, alpha, C, beta, Y, m, n, k);
}
