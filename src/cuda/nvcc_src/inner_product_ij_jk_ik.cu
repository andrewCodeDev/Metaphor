#include "../gemm_dispatch.cuh"

extern "C" void launch_inner_product_ij_jk_ik_Scalar(
  const void* a_data,
  const void* b_data,
  const double alpha,
        void* c_data,
  const double beta,
  len_t m, len_t n, len_t k,
  StreamContext stream
) {  
  // Metaphor is row major order, while cublas
  // is column major. To compute the correct
  // output, we need to perform T(B).T(A)
  // for the correct output combination
  mp_gemm<Scalar>::call(
    stream,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    k, m, n,
    alpha,
    b_data, k,
    a_data, n,
    beta,
    c_data, k
  );
}
