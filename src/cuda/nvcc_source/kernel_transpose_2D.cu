#include "../kernel_header.h"

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

__global__ void __kernel_transpose_2D_RScalar(
    const RScalar* src,
          RScalar* dst,
    RScalar dst_coef, // used in reverse
    len_t row,
    len_t col
){
    __shared__ RScalar tile[WARP_SIZE][WARP_SIZE + 1];
    len_t i_row = blockIdx.y * WARP_SIZE + threadIdx.y; // <- threadIdx.y in [0, 7)
    len_t i_col = blockIdx.x * WARP_SIZE + threadIdx.x; // <- threadIdx.x in [0, 32)

    // scale the matrix in tiles, 4 times for each thread,
    // note that the tiles are smaller than the block size
    // by a factor of 4

    for (len_t i = 0; i < WARP_SIZE; i += BLOCK_ROWS){
        if(i_col < col  && (i_row + i) < row){
            tile[threadIdx.y + i][threadIdx.x] = src[(i_row + i) * col + i_col];
        }
    }
    __syncthreads();

    i_row = blockIdx.x * WARP_SIZE + threadIdx.y;
    i_col = blockIdx.y * WARP_SIZE + threadIdx.x; 

    for (len_t i = 0; i < WARP_SIZE; i += BLOCK_ROWS){
        if(i_col < row  && (i_row + i) < col){
            
            const len_t i_dst = (i_row + i) * row + i_col;

            // blend output with destination forward dst_coef is 0, reverse is 1
            dst[i_dst] = tile[threadIdx.x][threadIdx.y + i] + (dst_coef * dst[i_dst]);
        }
    }
}

extern "C" void launch_transpose_2D_RScalar(
    Stream stream,
    const RScalar* src,
          RScalar* dst,
    RScalar dst_coef,
    len_t row,
    len_t col
) {
    const dim3 grid_block(
        DIMPAD(col, WARP_SIZE), 
        DIMPAD(row, WARP_SIZE)
    );

    const dim3 tile_block(
        WARP_SIZE, BLOCK_ROWS 
    );

    __kernel_transpose_2D_RScalar<<<grid_block, tile_block>>>(
        src, dst, dst_coef, row, col
    );
}
