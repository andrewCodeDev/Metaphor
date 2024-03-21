#include "../kernel_header.h"

__global__ void __kernel_softmax_ij_j_RScalar(
    const RScalar* A, 
          RScalar* B, 
    len_t m,
    len_t n
) {
  __shared__ RScalar redux[WARP_SIZE];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;

  const len_t offset = n * ((blockIdx.y * WARP_SIZE) + t_row);
  A += offset;
  B += offset;

  const len_t m_bound = (blockIdx.y * WARP_SIZE) + t_row;

  // these are mobile pointers
  auto A_tmp = A;
  auto B_tmp = B;

  //////////////////////////////////
  /// Grid Max /////////////////////

  RScalar grid_max = -Init<RScalar>::infinity();
    
  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      grid_max = MaxOP::apply(grid_max, A_tmp[t_col]);
    }
    A_tmp += WARP_SIZE;
  }

  grid_max = warpReduce<MaxOP>(grid_max);

  if (m_bound < m && t_col == 0) {
    redux[t_row] = grid_max;
  }

  __syncthreads();

  if (m_bound < m) {
    grid_max = redux[t_row];
  }

  //////////////////////////////////
  /// Grid Sum /////////////////////

  RScalar grid_sum = RScalar(0.0f);

  A_tmp = A;
    
  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      const RScalar u = rexp(A_tmp[t_col] - grid_max);
      B_tmp[t_col] = u;
      grid_sum += u;
    }
    A_tmp += WARP_SIZE;
    B_tmp += WARP_SIZE;
  }

  grid_sum = warpReduce<AddOP>(grid_sum);

  if (m_bound < m && t_col == 0) {
    redux[t_row] = grid_sum;
  }

  __syncthreads();

  if (m_bound < m) {
    grid_sum = redux[t_row];
  }
  
  for (len_t step = 0; step < n; step += WARP_SIZE) {

    if ((m_bound < m) && (step + t_col) < n) {    
      B[t_col] /= grid_sum;
    }
    // move A along the columns
    B += WARP_SIZE;
  }
} 

extern "C" void launch_softmax_ij_j_RScalar(
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

  __kernel_softmax_ij_j_RScalar<<<grid, block, 0, getCtx(stream)>>>(
    A, B, m, n
  );
}
