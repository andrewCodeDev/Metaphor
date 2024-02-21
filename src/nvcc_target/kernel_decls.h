/* GENERATED FILE */

#include "../tensor_types.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

EXTERN_C void launch_tanh_r16(
  Stream stream,
  const r16* a,
        r16* b, 
  len_t N
);
EXTERN_C void launch_tanh_r32(
  Stream stream,
  const r32* a,
        r32* b, 
  len_t N
);
EXTERN_C void launch_tanh_r64(
  Stream stream,
  const r64* a,
        r64* b, 
  len_t N
);
EXTERN_C void launch_hadamard_reverse_r16(
  Stream stream,
  r16 *grads_a,
  const r16 *value_b,
  const r16 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c16(
  Stream stream,
  c16 *grads_a,
  const c16 *value_b,
  const c16 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_r32(
  Stream stream,
  r32 *grads_a,
  const r32 *value_b,
  const r32 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c32(
  Stream stream,
  c32 *grads_a,
  const c32 *value_b,
  const c32 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_r64(
  Stream stream,
  r64 *grads_a,
  const r64 *value_b,
  const r64 *grads_c,
  len_t N
);
EXTERN_C void launch_hadamard_reverse_c64(
  Stream stream,
  c64 *grads_a,
  const c64 *value_b,
  const c64 *grads_c,
  len_t N
);
EXTERN_C void launch_subtraction_r16(
  Stream stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c16(
  Stream stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_subtraction_r32(
  Stream stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c32(
  Stream stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_subtraction_r64(
  Stream stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_subtraction_c64(
  Stream stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
EXTERN_C void launch_leaky_relu_r16(
  Stream stream,
  const r16* a,
        r16* b, 
        r16 coef,
  len_t N
);
EXTERN_C void launch_leaky_relu_r32(
  Stream stream,
  const r32* a,
        r32* b, 
        r32 coef,
  len_t N
);
EXTERN_C void launch_leaky_relu_r64(
  Stream stream,
  const r64* a,
        r64* b, 
        r64 coef,
  len_t N
);
EXTERN_C void launch_fill_r16(
  Stream stream,
  r16* dev_a,
  r16 value, 
  len_t N
);
EXTERN_C void launch_fill_c16(
  Stream stream,
  c16* dev_a,
  c16 value, 
  len_t N
);
EXTERN_C void launch_fill_r32(
  Stream stream,
  r32* dev_a,
  r32 value, 
  len_t N
);
EXTERN_C void launch_fill_c32(
  Stream stream,
  c32* dev_a,
  c32 value, 
  len_t N
);
EXTERN_C void launch_fill_r64(
  Stream stream,
  r64* dev_a,
  r64 value, 
  len_t N
);
EXTERN_C void launch_fill_c64(
  Stream stream,
  c64* dev_a,
  c64 value, 
  len_t N
);
EXTERN_C void launch_perumutate_r16(
  RTensor16 X, RTensor16 Y, Permutation P
);
EXTERN_C void launch_permutate_c16(
  CTensor16 X, CTensor16 Y, Permutation P
);
EXTERN_C void launch_perumutate_r32(
  RTensor32 X, RTensor32 Y, Permutation P
);
EXTERN_C void launch_permutate_c32(
  CTensor32 X, CTensor32 Y, Permutation P
);
EXTERN_C void launch_perumutate_r64(
  RTensor64 X, RTensor64 Y, Permutation P
);
EXTERN_C void launch_permutate_c64(
  CTensor64 X, CTensor64 Y, Permutation P
);
EXTERN_C void launch_relu_leaky_reverse_r16(
  Stream stream,
  const r16 *a_value,
        r16 *a_grads,
  const r16 *b_grads,
        r16 coef,
  len_t N
);
EXTERN_C void launch_relu_leaky_reverse_r32(
  Stream stream,
  const r32 *a_value,
        r32 *a_grads,
  const r32 *b_grads,
        r32 coef,
  len_t N
);
EXTERN_C void launch_relu_leaky_reverse_r64(
  Stream stream,
  const r64 *a_value,
        r64 *a_grads,
  const r64 *b_grads,
        r64 coef,
  len_t N
);
EXTERN_C void launch_addition_r16(
  Stream stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_addition_c16(
  Stream stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_addition_r32(
  Stream stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_addition_c32(
  Stream stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_addition_r64(
  Stream stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_addition_c64(
  Stream stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
EXTERN_C void launch_addition_reverse_r16(
  Stream stream,
  r16* a, 
  const r16* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c16(
  Stream stream,
  c16* a, 
  const c16* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_r32(
  Stream stream,
  r32* a, 
  const r32* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c32(
  Stream stream,
  c32* a, 
  const c32* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_r64(
  Stream stream,
  r64* a, 
  const r64* b, 
  len_t N
);
EXTERN_C void launch_addition_reverse_c64(
  Stream stream,
  c64* a, 
  const c64* b, 
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r16(
  Stream stream,
  r16* a, 
  const r16* b, 
  const r16 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c16(
  Stream stream,
  c16* a, 
  const c16* b, 
  const r16 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r32(
  Stream stream,
  r32* a, 
  const r32* b, 
  const r32 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c32(
  Stream stream,
  c32* a, 
  const c32* b, 
  const r32 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_r64(
  Stream stream,
  r64* a, 
  const r64* b, 
  const r64 coef,
  len_t N
);
EXTERN_C void launch_subtraction_reverse_c64(
  Stream stream,
  c64* a, 
  const c64* b, 
  const r64 coef,
  len_t N
);
EXTERN_C void launch_tanh_reverse_r16(
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
);
EXTERN_C void launch_tanh_reverse_r32(
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
);
EXTERN_C void launch_tanh_reverse_r64(
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
);
EXTERN_C void launch_hadamard_r16(
  Stream stream,
  const r16* a,
  const r16* b, 
  r16* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c16(
  Stream stream,
  const c16* a,
  const c16* b, 
  c16* c, 
  len_t N
);
EXTERN_C void launch_hadamard_r32(
  Stream stream,
  const r32* a,
  const r32* b, 
  r32* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c32(
  Stream stream,
  const c32* a,
  const c32* b, 
  c32* c, 
  len_t N
);
EXTERN_C void launch_hadamard_r64(
  Stream stream,
  const r64* a,
  const r64* b, 
  r64* c, 
  len_t N
);
EXTERN_C void launch_hadamard_c64(
  Stream stream,
  const c64* a,
  const c64* b, 
  c64* c, 
  len_t N
);
