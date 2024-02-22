#include "../kernel_header.h"

__global__ void __kernel_square_pow2_transpose_RScalar(
   const RScalar *src,
         RScalar *dst,
   len_t N
){
  // plus 1 reduced bank conflicts dramatically
  __shared__ RScalar tile[WARP_SIZE][WARP_SIZE + 1];

  len_t x = blockIdx.x * WARP_SIZE + threadIdx.x;
  len_t y = blockIdx.y * WARP_SIZE + threadIdx.y;
  len_t width = gridDim.x * WARP_SIZE;

  for (len_t j = 0; j < WARP_SIZE; j += N)
     tile[threadIdx.y+j][threadIdx.x] = src[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * WARP_SIZE + threadIdx.x;  // transpose block offset
  y = blockIdx.x * WARP_SIZE + threadIdx.y;

  for (len_t j = 0; j < WARP_SIZE; j += N)
     dst[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


__global__ void __kernel_square_pow2_transposeSCalar(
   const CScalar *src,
         CScalar *dst,
   len_t N
){
  // plus 1 reduced bank conflicts dramatically
  __shared__ CScalar tile[WARP_SIZE][WARP_SIZE + 1];

  len_t x = blockIdx.x * WARP_SIZE + threadIdx.x;
  len_t y = blockIdx.y * WARP_SIZE + threadIdx.y;
  len_t width = gridDim.x * WARP_SIZE;

  for (len_t j = 0; j < WARP_SIZE; j += N)
     tile[threadIdx.y+j][threadIdx.x] = src[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * WARP_SIZE + threadIdx.x;  // transpose block offset
  y = blockIdx.x * WARP_SIZE + threadIdx.y;

  for (len_t j = 0; j < WARP_SIZE; j += N)
     dst[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}
