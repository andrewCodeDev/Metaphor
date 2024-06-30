#ifndef MP_GEMM_HEADER
#define MP_GEMM_HEADER

#include "kernel_header.h"

template<class T> struct mp_gemm;

template<> struct mp_gemm<r16> {

  using type = r16;

  static void call(
      StreamContext stream,
      cublasOperation_t trans_a,  
      cublasOperation_t trans_b,  
      len_t m, len_t n, len_t k,
      type alpha,
      const void* a_data, len_t lda,
      const void* b_data, len_t ldb,
      type beta,
            void* c_data, len_t ldc
      
  ) {
    CUBLAS_ASSERT(cublasHgemm(
          get_handle(stream), 
          trans_a,
          trans_b, 
          static_cast<int32_t>(m),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          &alpha, 
          static_cast<const type*>(a_data), 
          static_cast<int32_t>(lda),
          static_cast<const type*>(b_data), 
          static_cast<int32_t>(ldb),
          &beta, 
          static_cast<type*>(c_data), 
          static_cast<int32_t>(ldc)
    ));
  }
};

template<> struct mp_gemm<r32> {

  using type = r32;

  static void call(
      StreamContext stream,
      cublasOperation_t trans_a,  
      cublasOperation_t trans_b,  
      len_t m, len_t n, len_t k,
      type alpha,
      const void* a_data, len_t lda,
      const void* b_data, len_t ldb,
      type beta,
            void* c_data, len_t ldc
      
  ) {
    CUBLAS_ASSERT(cublasSgemm(
          get_handle(stream), 
          trans_a,
          trans_b, 
          static_cast<int32_t>(m),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          &alpha, 
          static_cast<const type*>(a_data), 
          static_cast<int32_t>(lda),
          static_cast<const type*>(b_data), 
          static_cast<int32_t>(ldb),
          &beta, 
          static_cast<type*>(c_data), 
          static_cast<int32_t>(ldc)
    ));
  }
};


template<> struct mp_gemm<r64> {

  using type = r64;

  static void call(
      StreamContext stream,
      cublasOperation_t trans_a,  
      cublasOperation_t trans_b,  
      len_t m, len_t n, len_t k,
      type alpha,
      const void* a_data, len_t lda,
      const void* b_data, len_t ldb,
      type beta,
            void* c_data, len_t ldc
      
  ) {
    CUBLAS_ASSERT(cublasDgemm(
          get_handle(stream), 
          trans_a,
          trans_b, 
          static_cast<int32_t>(m),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          &alpha, 
          static_cast<const type*>(a_data), 
          static_cast<int32_t>(lda),
          static_cast<const type*>(b_data), 
          static_cast<int32_t>(ldb),
          &beta, 
          static_cast<type*>(c_data), 
          static_cast<int32_t>(ldc)
    ));
  }
};

#endif

