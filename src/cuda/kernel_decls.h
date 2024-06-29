/* GENERATED FILE */

#include "kernel_header.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

EXTERN_C void launch_reduce_ij_i_r16(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_reduce_ij_i_r32(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_reduce_ij_i_r64(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_reduce_ij_j_r16(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_reduce_ij_j_r32(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_reduce_ij_j_r64(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_broadcast_i_ij_r16(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_broadcast_i_ij_r32(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_broadcast_i_ij_r64(
    const void* src,
          void* dst,
    double alpha,
    len_t m,
    len_t n,
    StreamContext stream
);
EXTERN_C void launch_dilate_reverse_r16(
  const void* a,
  double value, 
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_dilate_reverse_r32(
  const void* a,
  double value, 
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_dilate_reverse_r64(
  const void* a,
  double value, 
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_subtraction_r16(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_subtraction_r32(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_subtraction_r64(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_relu_reverse_r16(
  const void *a_value,
        void *a_grads,
  const void *b_grads,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_relu_reverse_r32(
  const void *a_value,
        void *a_grads,
  const void *b_grads,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_relu_reverse_r64(
  const void *a_value,
        void *a_grads,
  const void *b_grads,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_dilate_r16(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_dilate_r32(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_dilate_r64(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_translate_r16(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_translate_r32(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_translate_r64(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_negate_r16(
  const void* a,
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_negate_r32(
  const void* a,
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_negate_r64(
  const void* a,
        void* b, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_reverse_r16(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_reverse_r32(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_reverse_r64(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_r16(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_r32(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_hadamard_r64(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_fill_r16(
  void* dev_a,
  double value, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_fill_r32(
  void* dev_a,
  double value, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_fill_r64(
  void* dev_a,
  double value, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_relu_r16(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_relu_r32(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_relu_r64(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_addition_r16(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_addition_r32(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_addition_r64(
  const void* a,
  const void* b, 
  void* c, 
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_broadcast_j_ij_r16(
  const void* src, 
        void* dst, 
  double alpha,
  len_t m,
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_broadcast_j_ij_r32(
  const void* src, 
        void* dst, 
  double alpha,
  len_t m,
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_broadcast_j_ij_r64(
  const void* src, 
        void* dst, 
  double alpha,
  len_t m,
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_copy_r16(
  const r16* src, 
        r16* dst, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_copy_r32(
  const r32* src, 
        r32* dst, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_copy_r64(
  const r64* src, 
        r64* dst, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_sequence_r16(
  void* dev_a,
  double init,
  double step,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_sequence_r32(
  void* dev_a,
  double init,
  double step,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_sequence_r64(
  void* dev_a,
  double init,
  double step,
  len_t N,
  StreamContext stream
);
EXTERN_C void launch_stepwise_r16(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_stepwise_r32(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
EXTERN_C void launch_stepwise_r64(
  const void* a,
        void* b, 
  len_t n,
  StreamContext stream
);
