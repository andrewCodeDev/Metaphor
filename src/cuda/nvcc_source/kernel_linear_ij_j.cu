#include "../kernel_header.h"

__global__ void __kernel_inner_product_ij_j_RScalar(
  const RScalar *x,
  const RScalar *A, 
        RScalar alpha, // scales product
  const RScalar *b,
        RScalar beta, // blends y back in
        RScalar *y,
  len_t M, 
  len_t N
) {
  constexpr len_t WARP_STEP = coalesce<RScalar>::warp_step;
  
  // check for bank conflicts... the idea is to coalesce
  // these values, but that may be slower than unrolling
  __shared__ RScalar tile_A[WARP_SIZE][WARP_SIZE];
  __shared__ RScalar output[WARP_SIZE][WARP_STEP];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;
  
  // this boundary doesn't change, calculate once
  const len_t m_pos = blockIdx.y * WARP_SIZE + t_row;

  // this is here for cases of small N
  const len_t min_n = min(N, WARP_SIZE);

  // locate the correct row on matrix A
  A += (N * t_row) + (blockIdx.y * WARP_SIZE * N);

  // locate the correct row on vector y
  y += (blockIdx.y * WARP_SIZE);

  b += (blockIdx.y * WARP_SIZE);

  // reload our warp tile and coalesce tile values
  auto tc = reinterpret_cast<coalesce<RScalar>::c_ptr>(tile_A[t_row]);

  // advance through our warp tiles and coalesce x
  auto xc = reinterpret_cast<coalesce<RScalar>::c_ptr>(x);
  
  RScalar out = RScalar(0.0f);

  // initialize shared memory to zero
  tile_A[t_row][t_col] = out;

  for (len_t step = 0; step < N; step += WARP_SIZE) {

    // check against the row of A
    const len_t n_pos_A = t_col + step;

    // only load if inbounds of A
    if ((m_pos < M) && (n_pos_A < N))
      tile_A[t_row][t_col] = A[t_col];

    __syncthreads();

    if (m_pos < M && t_col < WARP_STEP) {

      // make sure our calculation is in-bounds
      const len_t stop = min(min_n, N - step);

      const len_t idx = t_col * 4;

      // we have another warp
      if ((stop == WARP_SIZE) || ((idx + 4) < stop)) {
        const auto u = tc[t_col];
        const auto v = xc[t_col];
        out += u.w * v.w;
        out += u.x * v.x;
        out += u.y * v.y;
        out += u.z * v.z;

      } else { // remainder

        // load what you can in first/last block
        for (len_t i = idx, j = (step + idx); j < N; ++i, ++j) { 
          out += tile_A[t_row][i] * x[j]; 
        }
      }
    }

    // shift tile over the cols of A
    A += WARP_SIZE;

    // shift coalesced one warp step
    xc += WARP_STEP;

    __syncthreads();
  }

  // load in helper thread values
  if (t_col < WARP_STEP && m_pos < M) {
    output[t_row][t_col] = out;
  }

  __syncthreads();

  if (t_col == 0 && m_pos < M) {

    // sum helper thread results
    for (len_t i = 1; i < WARP_STEP; ++i) {
      out += output[t_row][i];
    }

    y[t_row] = out * alpha + b[t_row] * beta;
  }
}

extern "C" void launch_inner_product_ij_j_RScalar(
  StreamCtx stream,
  const RScalar *x,
  const RScalar *A, 
        RScalar alpha, // scales product
  const RScalar *b,
        RScalar beta, // blends y back in
        RScalar *y,
  len_t M, 
  len_t N
) {
    dim3 grid_block(
        1, DIMPAD(M, WARP_SIZE)
    );

    dim3 tile_block(
        WARP_SIZE,
        WARP_SIZE
    );

    __kernel_inner_product_ij_j_RScalar
        <<<grid_block, tile_block, 0, getCtx(stream)>>>(A, x, alpha, b, beta, y, M, N);
}
