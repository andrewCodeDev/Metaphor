/* GENERATED FILE */

#include "kernel_header.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

EXTERN_C void launch_permutate_ij_ji_r16(
    StreamCtx stream,
    const r16* src,
          r16* dst,
    r16 dst_coef,
    len_t row,
    len_t col
);
EXTERN_C void launch_permutate_ij_ji_r32(
    StreamCtx stream,
    const r32* src,
          r32* dst,
    r32 dst_coef,
    len_t row,
    len_t col
);
EXTERN_C void launch_permutate_ij_ji_r64(
    StreamCtx stream,
    const r64* src,
          r64* dst,
    r64 dst_coef,
    len_t row,
    len_t col
);
EXTERN_C void launch_logistic_reverse_r16(
  StreamCtx stream,
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
);
EXTERN_C void launch_logistic_reverse_r32(
  StreamCtx stream,
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
);
EXTERN_C void launch_logistic_reverse_r64(
  StreamCtx stream,
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
);
EXTERN_C void launch_tanh_r16(
  StreamCtx stream,
  const r16* a,
        r16* b, 
  len_t N
);
EXTERN_C void launch_tanh_r32(
  StreamCtx stream,
  const r32* a,
        r32* b, 
  len_t N
);
EXTERN_C void launch_tanh_r64(
  StreamCtx stream,
  const r64* a,
        r64* b, 
  len_t N
);
EXTERN_C void launch_cce_loss_i_i_r16(
  StreamCtx stream,
  const r16* src_value, 
        r16* src_grads, 
        len_t    trg,
        r16* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_cce_loss_i_i_r32(
  StreamCtx stream,
  const r32* src_value, 
        r32* src_grads, 
        len_t    trg,
        r32* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_cce_loss_i_i_r64(
  StreamCtx stream,
  const r64* src_value, 
        r64* src_grads, 
        len_t    trg,
        r64* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_hadamard_reverse_r16(
  StreamCtx stream,
  r16 *grads_a,
  const r16 *value_b,
  const r16 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c16(
  StreamCtx stream,
  c16 *grads_a,
  const c16 *value_b,
  const c16 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_r32(
  StreamCtx stream,
  r32 *grads_a,
  const r32 *value_b,
  const r32 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c32(
  StreamCtx stream,
  c32 *grads_a,
  const c32 *value_b,
  const c32 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_r64(
  StreamCtx stream,
  r64 *grads_a,
  const r64 *value_b,
  const r64 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c64(
  StreamCtx stream,
  c64 *grads_a,
  const c64 *value_b,
  const c64 *grads_c,
  len_t N
);
EXTERN_C void launch_outer_product_i_j_r16(
  StreamCtx stream,
  const r16 *x,
  const r16 *y, 
        r16 alpha, // scales product
        r16 *A,
        r16 beta, // blends A back in
  len_t M, 
  len_t N
);
EXTERN_C void launch_outer_product_i_j_r32(
  StreamCtx stream,
  const r32 *x,
  const r32 *y, 
        r32 alpha, // scales product
        r32 *A,
        r32 beta, // blends A back in
  len_t M, 
  len_t N
);
EXTERN_C void launch_outer_product_i_j_r64(
  StreamCtx stream,
  const r64 *x,
  const r64 *y, 
        r64 alpha, // scales product
        r64 *A,
        r64 beta, // blends A back in
  len_t M, 
  len_t N
);
EXTERN_C void launch_subtraction_r16(
  StreamCtx stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c16(
  StreamCtx stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_subtraction_r32(
  StreamCtx stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c32(
  StreamCtx stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_subtraction_r64(
  StreamCtx stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c64(
  StreamCtx stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
EXTERN_C void launch_linear_ij_jk_r16(
  StreamCtx stream,
  const r16 *A, 
  const r16 *B,
        r16 alpha, // scales product
  const r16 *C,
        r16 beta, // blends C back in
        r16 *Y,
  len_t m, 
  len_t n, 
  len_t k 
);
EXTERN_C void launch_linear_ij_jk_r32(
  StreamCtx stream,
  const r32 *A, 
  const r32 *B,
        r32 alpha, // scales product
  const r32 *C,
        r32 beta, // blends C back in
        r32 *Y,
  len_t m, 
  len_t n, 
  len_t k 
);
EXTERN_C void launch_linear_ij_jk_r64(
  StreamCtx stream,
  const r64 *A, 
  const r64 *B,
        r64 alpha, // scales product
  const r64 *C,
        r64 beta, // blends C back in
        r64 *Y,
  len_t m, 
  len_t n, 
  len_t k 
);
EXTERN_C void launch_linear_i_ij_r16(
  StreamCtx stream,
  const r16 *x,
  const r16 *A, 
        r16 alpha, // scales product
  const r16 *b,
        r16 beta, // blends y back in
        r16 *y,
  len_t M, 
  len_t N
);
EXTERN_C void launch_linear_i_ij_r32(
  StreamCtx stream,
  const r32 *x,
  const r32 *A, 
        r32 alpha, // scales product
  const r32 *b,
        r32 beta, // blends y back in
        r32 *y,
  len_t M, 
  len_t N
);
EXTERN_C void launch_linear_i_ij_r64(
  StreamCtx stream,
  const r64 *x,
  const r64 *A, 
        r64 alpha, // scales product
  const r64 *b,
        r64 beta, // blends y back in
        r64 *y,
  len_t M, 
  len_t N
);
EXTERN_C void launch_softmax_i_i_r16(
  StreamCtx stream,
  const r16* A, 
        r16* B, 
        r16* scratch,
  len_t m
);
EXTERN_C void launch_softmax_i_i_r32(
  StreamCtx stream,
  const r32* A, 
        r32* B, 
        r32* scratch,
  len_t m
);
EXTERN_C void launch_softmax_i_i_r64(
  StreamCtx stream,
  const r64* A, 
        r64* B, 
        r64* scratch,
  len_t m
);
EXTERN_C void launch_selu_reverse_r16(
  StreamCtx stream,
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
);
EXTERN_C void launch_selu_reverse_r32(
  StreamCtx stream,
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
);
EXTERN_C void launch_selu_reverse_r64(
  StreamCtx stream,
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
);
EXTERN_C void launch_leaky_relu_r16(
  StreamCtx stream,
  const r16* a,
        r16* b, 
        r16 coef,
  len_t N
);
EXTERN_C void launch_leaky_relu_r32(
  StreamCtx stream,
  const r32* a,
        r32* b, 
        r32 coef,
  len_t N
);
EXTERN_C void launch_leaky_relu_r64(
  StreamCtx stream,
  const r64* a,
        r64* b, 
        r64 coef,
  len_t N
);
EXTERN_C void launch_mse_loss_i_i_r16(
  StreamCtx stream,
  const r16* src_value, 
        r16* src_grads, 
  const r16* trg_value, 
        r16* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_mse_loss_i_i_r32(
  StreamCtx stream,
  const r32* src_value, 
        r32* src_grads, 
  const r32* trg_value, 
        r32* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_mse_loss_i_i_r64(
  StreamCtx stream,
  const r64* src_value, 
        r64* src_grads, 
  const r64* trg_value, 
        r64* scratch,
        double*  redux, // scalar
  len_t m
);
EXTERN_C void launch_reduce_key_ij_j_r16(
    StreamCtx stream,
    const r16* src,
          r16* dst,
    const unsigned* keys,
          r16 alpha,
          r16* scratch,
    len_t src_col,
    len_t key_len
);
EXTERN_C void launch_reduce_key_ij_j_r32(
    StreamCtx stream,
    const r32* src,
          r32* dst,
    const unsigned* keys,
          r32 alpha,
          r32* scratch,
    len_t src_col,
    len_t key_len
);
EXTERN_C void launch_reduce_key_ij_j_r64(
    StreamCtx stream,
    const r64* src,
          r64* dst,
    const unsigned* keys,
          r64 alpha,
          r64* scratch,
    len_t src_col,
    len_t key_len
);
EXTERN_C void launch_copy_indexed_ij_kj_r16(
    StreamCtx stream,
    const r16* src,
          r16* dst,
    const len_t* idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_copy_indexed_ij_kj_buffered_r16(
    StreamCtx stream,
    const r16* src,
          r16* dst,
    UintBuffer idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_copy_indexed_ij_kj_r32(
    StreamCtx stream,
    const r32* src,
          r32* dst,
    const len_t* idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_copy_indexed_ij_kj_buffered_r32(
    StreamCtx stream,
    const r32* src,
          r32* dst,
    UintBuffer idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_copy_indexed_ij_kj_r64(
    StreamCtx stream,
    const r64* src,
          r64* dst,
    const len_t* idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_copy_indexed_ij_kj_buffered_r64(
    StreamCtx stream,
    const r64* src,
          r64* dst,
    UintBuffer idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
);
EXTERN_C void launch_fill_r16(
  StreamCtx stream,
  r16* dev_a,
  r16 value, 
  len_t N
);
EXTERN_C void launch_fill_c16(
  StreamCtx stream,
  c16* dev_a,
  c16 value, 
  len_t N
);
EXTERN_C void launch_fill_r32(
  StreamCtx stream,
  r32* dev_a,
  r32 value, 
  len_t N
);
EXTERN_C void launch_fill_c32(
  StreamCtx stream,
  c32* dev_a,
  c32 value, 
  len_t N
);
EXTERN_C void launch_fill_r64(
  StreamCtx stream,
  r64* dev_a,
  r64 value, 
  len_t N
);
EXTERN_C void launch_fill_c64(
  StreamCtx stream,
  c64* dev_a,
  c64 value, 
  len_t N
);
EXTERN_C void launch_permutate_r16(
  RTensor16 X, RTensor16 Y, Permutation P
);
EXTERN_C void launch_permutate_naive_c16(
  CTensor16 X, CTensor16 Y, Permutation P
);
EXTERN_C void launch_permutate_r32(
  RTensor32 X, RTensor32 Y, Permutation P
);
EXTERN_C void launch_permutate_naive_c32(
  CTensor32 X, CTensor32 Y, Permutation P
);
EXTERN_C void launch_permutate_r64(
  RTensor64 X, RTensor64 Y, Permutation P
);
EXTERN_C void launch_permutate_naive_c64(
  CTensor64 X, CTensor64 Y, Permutation P
);
EXTERN_C void launch_relu_leaky_reverse_r16(
  StreamCtx stream,
  const r16 *a_value,
        r16 *a_grads,
  const r16 *b_grads,
        r16 coef,
  len_t N
);
EXTERN_C void launch_relu_leaky_reverse_r32(
  StreamCtx stream,
  const r32 *a_value,
        r32 *a_grads,
  const r32 *b_grads,
        r32 coef,
  len_t N
);
EXTERN_C void launch_relu_leaky_reverse_r64(
  StreamCtx stream,
  const r64 *a_value,
        r64 *a_grads,
  const r64 *b_grads,
        r64 coef,
  len_t N
);
EXTERN_C void launch_addition_r16(
  StreamCtx stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_addition_c16(
  StreamCtx stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_addition_r32(
  StreamCtx stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_addition_c32(
  StreamCtx stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_addition_r64(
  StreamCtx stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_addition_c64(
  StreamCtx stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
EXTERN_C void launch_softmax_ij_j_r16(
  StreamCtx stream,
  const r16* A, 
        r16* B, 
  len_t m,
  len_t n
);
EXTERN_C void launch_softmax_ij_j_r32(
  StreamCtx stream,
  const r32* A, 
        r32* B, 
  len_t m,
  len_t n
);
EXTERN_C void launch_softmax_ij_j_r64(
  StreamCtx stream,
  const r64* A, 
        r64* B, 
  len_t m,
  len_t n
);
EXTERN_C void launch_softmax_i_i_reverse_r16(
  StreamCtx stream,
        r16* a_grads, 
  const r16* b_value, 
  const r16* b_grads,
        r16* scratch,
  len_t m
);
EXTERN_C void launch_softmax_i_i_reverse_r32(
  StreamCtx stream,
        r32* a_grads, 
  const r32* b_value, 
  const r32* b_grads,
        r32* scratch,
  len_t m
);
EXTERN_C void launch_softmax_i_i_reverse_r64(
  StreamCtx stream,
        r64* a_grads, 
  const r64* b_value, 
  const r64* b_grads,
        r64* scratch,
  len_t m
);
EXTERN_C void launch_addition_reverse_r16(
  StreamCtx stream,
  r16* a, 
  const r16* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c16(
  StreamCtx stream,
  c16* a, 
  const c16* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_r32(
  StreamCtx stream,
  r32* a, 
  const r32* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c32(
  StreamCtx stream,
  c32* a, 
  const c32* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_r64(
  StreamCtx stream,
  r64* a, 
  const r64* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c64(
  StreamCtx stream,
  c64* a, 
  const c64* b, 
  len_t N
);
EXTERN_C void launch_cce_loss_ij_j_r16(
  StreamCtx stream,
  const r16* src_value, 
        r16* src_grads, 
  const len_t* trgs,
        r16* scratch,
        double* redux, // scalar
  len_t m,
  len_t n
);
EXTERN_C void launch_cce_loss_ij_j_r32(
  StreamCtx stream,
  const r32* src_value, 
        r32* src_grads, 
  const len_t* trgs,
        r32* scratch,
        double* redux, // scalar
  len_t m,
  len_t n
);
EXTERN_C void launch_cce_loss_ij_j_r64(
  StreamCtx stream,
  const r64* src_value, 
        r64* src_grads, 
  const len_t* trgs,
        r64* scratch,
        double* redux, // scalar
  len_t m,
  len_t n
);
EXTERN_C void launch_sequence_r16(
  StreamCtx stream,
  r16* dev_a,
  r16 init,
  r16 step,
  len_t N
);
EXTERN_C void launch_sequence_r32(
  StreamCtx stream,
  r32* dev_a,
  r32 init,
  r32 step,
  len_t N
);
EXTERN_C void launch_sequence_r64(
  StreamCtx stream,
  r64* dev_a,
  r64 init,
  r64 step,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r16(
  StreamCtx stream,
  r16* a, 
  const r16* b, 
  const r16 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c16(
  StreamCtx stream,
  c16* a, 
  const c16* b, 
  const r16 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r32(
  StreamCtx stream,
  r32* a, 
  const r32* b, 
  const r32 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c32(
  StreamCtx stream,
  c32* a, 
  const c32* b, 
  const r32 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r64(
  StreamCtx stream,
  r64* a, 
  const r64* b, 
  const r64 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c64(
  StreamCtx stream,
  c64* a, 
  const c64* b, 
  const r64 coef,
  len_t N
);
EXTERN_C void launch_gradient_descent_r16(
  StreamCtx stream,
        r16* a_value,
  const r16* a_grads, 
  r16 rate,
  r16 lower,
  r16 upper,
  len_t N
);
EXTERN_C void launch_gradient_descent_r32(
  StreamCtx stream,
        r32* a_value,
  const r32* a_grads, 
  r32 rate,
  r32 lower,
  r32 upper,
  len_t N
);
EXTERN_C void launch_gradient_descent_r64(
  StreamCtx stream,
        r64* a_value,
  const r64* a_grads, 
  r64 rate,
  r64 lower,
  r64 upper,
  len_t N
);
EXTERN_C void launch_softmax_ij_j_reverse_r16(
  StreamCtx stream,
        r16* A_grads,
  const r16* B_value, 
  const r16* B_grads,
  len_t m,
  len_t n
);
EXTERN_C void launch_softmax_ij_j_reverse_r32(
  StreamCtx stream,
        r32* A_grads,
  const r32* B_value, 
  const r32* B_grads,
  len_t m,
  len_t n
);
EXTERN_C void launch_softmax_ij_j_reverse_r64(
  StreamCtx stream,
        r64* A_grads,
  const r64* B_value, 
  const r64* B_grads,
  len_t m,
  len_t n
);
EXTERN_C void launch_tanh_reverse_r16(
  StreamCtx stream,
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
);
EXTERN_C void launch_tanh_reverse_r32(
  StreamCtx stream,
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
);
EXTERN_C void launch_tanh_reverse_r64(
  StreamCtx stream,
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
);
EXTERN_C void launch_hadamard_r16(
  StreamCtx stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c16(
  StreamCtx stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_hadamard_r32(
  StreamCtx stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c32(
  StreamCtx stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_hadamard_r64(
  StreamCtx stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c64(
  StreamCtx stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
EXTERN_C void launch_selu_r16(
  StreamCtx stream,
  const r16* a,
        r16* b, 
  len_t N
);
EXTERN_C void launch_selu_r32(
  StreamCtx stream,
  const r32* a,
        r32* b, 
  len_t N
);
EXTERN_C void launch_selu_r64(
  StreamCtx stream,
  const r64* a,
        r64* b, 
  len_t N
);
EXTERN_C void launch_linear_ij_j_r16(
  StreamCtx stream,
  const r16 *A, 
  const r16 *x,
        r16 alpha, // scales product
  const r16 *b,
        r16 beta, // blends y back in
        r16 *y,
  len_t M, 
  len_t N
);
EXTERN_C void launch_linear_ij_j_r32(
  StreamCtx stream,
  const r32 *A, 
  const r32 *x,
        r32 alpha, // scales product
  const r32 *b,
        r32 beta, // blends y back in
        r32 *y,
  len_t M, 
  len_t N
);
EXTERN_C void launch_linear_ij_j_r64(
  StreamCtx stream,
  const r64 *A, 
  const r64 *x,
        r64 alpha, // scales product
  const r64 *b,
        r64 beta, // blends y back in
        r64 *y,
  len_t M, 
  len_t N
);
