#include "../kernel_header.h"

__global__ void __kernel_minmax_ij_j_reverse_RScalar(
  const RScalar* a_value, 
        RScalar* a_grads, 
  const RScalar* b_grads, 
  len_t m,
  len_t n
) {
  __shared__ RScalar redux[WARP_SIZE][2];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;

  const len_t offset = n * ((blockIdx.y * WARP_SIZE) + t_row);
  a_value += offset;
  a_grads += offset;
  b_grads += offset;

  const len_t m_bound = (blockIdx.y * WARP_SIZE) + t_row;

  // these are mobile pointers
  auto a_tmp = a_value;

  //////////////////////////////////
  /// Grid Max /////////////////////

  RScalar grid_min =  Init<RScalar>::infinity();
  RScalar grid_max = -Init<RScalar>::infinity();
    
  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      grid_min = MinOP::apply(grid_min, a_tmp[t_col]);
      grid_max = MaxOP::apply(grid_max, a_tmp[t_col]);
    }
    a_tmp += WARP_SIZE;
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
      a_grads[t_col] += b_grads[t_col] / (grid_max - grid_min);
    }
    // move A along the columns
    a_grads += WARP_SIZE;
    b_grads += WARP_SIZE;
  }
} 

extern "C" void launch_minmax_ij_j_reverse_RScalar(
  StreamCtx stream,
  const RScalar* a_value, 
        RScalar* a_grads, 
  const RScalar* b_grads, 
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

  __kernel_minmax_ij_j_reverse_RScalar<<<grid, block, 0, getCtx(stream)>>>(
    a_value, a_grads, b_grads, m, n
  );
}
