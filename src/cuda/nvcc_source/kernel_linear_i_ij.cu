#include "../kernel_header.h"

__global__ void __kernel_linear_i_ij_RScalar(
  const RScalar* x, // vector
  const RScalar* A, // matrix
  RScalar alpha, // scale x.A
  const RScalar* b, // bias
  RScalar beta,  // scale b
        RScalar* y,
  len_t M,
  len_t N
) {
  // (1, M) x (M, N) -> (1, N)
  
  constexpr len_t WARP_STEP = coalesce<RScalar>::warp_step;
  
  // check for bank conflicts... the idea is to coalesce
  // these values, but that may be slower than unrolling
  __shared__ RScalar tile_A[WARP_SIZE][WARP_SIZE];
  __shared__ RScalar output[WARP_SIZE][WARP_STEP];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;
  
  // this boundary doesn't change, calculate once
  const len_t n_pos = blockIdx.x * WARP_SIZE + t_col;

  // this boundary doesn't change, calculate once (transposed)
  const len_t n_pos_t = blockIdx.x * WARP_SIZE + t_row;

  // this is here for cases of small M
  const len_t min_m = min(M, WARP_SIZE);

  // shift A over the column space
  A += (blockIdx.x * WARP_SIZE) + (N * t_row);

  // locate the correct col on vector y
  y += (blockIdx.x * WARP_SIZE);

  b += (blockIdx.x * WARP_SIZE);

  // reload our warp tile and coalesce tile values
  auto tc = reinterpret_cast<coalesce<RScalar>::c_ptr>(tile_A[t_row]);

  // advance through our warp tiles and coalesce x
  auto xc = reinterpret_cast<coalesce<RScalar>::c_ptr>(x);

  RScalar out = RScalar(0.0f);

  // initialize shared memory to zero
  tile_A[t_row][t_col] = out;

  for (len_t step = 0; step < M; step += WARP_SIZE) {

    // check that we don't overrun M dimension
    const len_t m_pos = step + t_row;

    // only load transpose if inbounds of A
    if ((m_pos < M) && (n_pos < N))
      tile_A[t_col][t_row] = A[t_col];

    __syncthreads();

    // since we transposed, m_pos now refers to calculation cols
    if ((n_pos_t < N) && (t_col < WARP_STEP)) {

      // make sure our calculation is in-bounds
      const len_t stop = min(min_m, M - step);

      const len_t idx = t_col * 4;

      // we have another warp
      if (stop == WARP_SIZE || ((idx + 4) < stop)) {
        const auto u = tc[t_col];
        const auto v = xc[t_col];
        out += u.w * v.w;
        out += u.x * v.x;
        out += u.y * v.y;
        out += u.z * v.z;

      } else { // remainder

        // load what you can in first/last block
        for (len_t i = idx, j = (step + idx); j < M; ++i, ++j)
          out += tile_A[t_row][i] * x[j];         
      }
    }

    // shift tile over the rows of A
    A += WARP_SIZE * N;

    // shift coalesced one warp step
    xc += WARP_STEP;

    __syncthreads();
  }

  // load in helper thread values
  if ((n_pos_t < N) && (t_col < WARP_STEP)) {
    output[t_row][t_col] = out;
  }

  __syncthreads();

  if ((n_pos < N) && (t_col == 0)) {

    // sum helper thread results
    for (len_t i = 1; i < WARP_STEP; ++i) {
      out += output[t_row][i];
    }

    // write results out

    y[t_row] = out * alpha + b[t_row] * beta;
  }
}

extern "C" void launch_linear_i_ij_RScalar(
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
        DIMPAD(N, WARP_SIZE), 1
    );

    dim3 tile_block(
        WARP_SIZE,
        WARP_SIZE
    );

    __kernel_linear_i_ij_RScalar
        <<<grid_block, tile_block, 0, getCtx(stream)>>>(x, A, alpha, b, beta, y, M, N);
}
