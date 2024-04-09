#include "../kernel_header.h"

__global__ void __kernel_minmax_ij_j_RScalar(
    const RScalar* A, 
          RScalar* B, 
    len_t m,
    len_t n
) {
  __shared__ RScalar redux[WARP_SIZE][2];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;

  const len_t offset = n * ((blockIdx.y * WARP_SIZE) + t_row);
  A += offset;
  B += offset;

  const len_t m_bound = (blockIdx.y * WARP_SIZE) + t_row;

  //////////////////////////////////
  /// Grid Max /////////////////////

  RScalar grid_min =  Init<RScalar>::infinity();
  RScalar grid_max = -Init<RScalar>::infinity();
    
  auto A_tmp = A;

  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      grid_min = MinOP::apply(grid_min, A_tmp[t_col]);
      grid_max = MaxOP::apply(grid_max, A_tmp[t_col]);
    }
    A_tmp += WARP_SIZE;
  }

  grid_min = warpReduce<MinOP>(grid_min);
  grid_max = warpReduce<MaxOP>(grid_max);

  if (m_bound < m && t_col == 0) {
    redux[t_row][0] = grid_min;
    redux[t_row][1] = grid_max;
  }

  __syncthreads();

  if (m_bound < m) {
    grid_min = redux[t_row][0];
    grid_max = redux[t_row][1];
  }

  //////////////////////////////////
  /// Grid Sum /////////////////////

  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      B[t_col] = (A[t_col] - grid_min) / (grid_max - grid_min);
    }
    // move A along the columns
    A += WARP_SIZE;
    B += WARP_SIZE;
  }
} 

extern "C" void launch_minmax_ij_j_RScalar(
  StreamCtx stream,
  const RScalar* A, 
        RScalar* B, 
  len_t m,
  len_t n
) {

  dim3 grid (
    1, DIMPAD(m, WARP_SIZE)
  );

  dim3 block (
    WARP_SIZE, 
    WARP_SIZE
  );

  __kernel_minmax_ij_j_RScalar<<<grid, block, 0, getCtx(stream)>>>(
    A, B, m, n
  );
}
