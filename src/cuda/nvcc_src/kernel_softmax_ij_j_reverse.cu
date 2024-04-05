#include "../kernel_header.h"

__global__ void __kernel_softmax_ij_j_reverse_RScalar(
          RScalar* A_grads,
    const RScalar* B_value, 
    const RScalar* B_grads, 
    len_t m,
    len_t n
) {
  __shared__ RScalar redux[WARP_SIZE];

  // find our tile row and column
  len_t t_row = threadIdx.y;
  len_t t_col = threadIdx.x;

  const len_t m_bound = blockIdx.y * blockDim.y + t_row;

  A_grads += m_bound * n;
  B_value += m_bound * n;
  B_grads += m_bound * n;

  //////////////////////////////////
  /// Grid Sum /////////////////////

  auto B_val = B_value;
  auto B_grd = B_grads;

  RScalar grid_sum = RScalar(0.0f);

  for (len_t step = 0; step < m; step += blockDim.x) {

    if ((m_bound < m) && (step + t_col) < n) {    
      grid_sum += B_val[t_col] * B_grd[t_col];
    }
    // move A along the columns
    B_val += blockDim.x;
    B_grd += blockDim.x;
  }

  grid_sum = warpReduce<AddOP>(grid_sum);

  if (m_bound < m && t_col == 0) {
    redux[t_row] = grid_sum;
  }

  __syncthreads();

  if (m_bound < m) {
    grid_sum = redux[t_row];
  }
  
  for (len_t step = 0; step < n; step += blockDim.x) {

    if ((m_bound < m) && (step + t_col) < n) {    
      A_grads[t_col] = B_value[t_col] * (B_grads[t_col] - grid_sum);
    }
    // move A along the columns
    A_grads += blockDim.x;
    B_value += blockDim.x;
    B_grads += blockDim.x;
  }
} 

extern "C" void launch_softmax_ij_j_reverse_RScalar(
  StreamCtx stream,
        RScalar* A_grads,
  const RScalar* B_value, 
  const RScalar* B_grads,
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

  __kernel_softmax_ij_j_reverse_RScalar<<<grid, block, 0, getCtx(stream)>>>(
    A_grads, B_value, B_grads, m, n
  );
}
