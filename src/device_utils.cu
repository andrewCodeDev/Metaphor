
#include "device_utils.h"

#include "kernel_header.h"

extern "C" void mpMemAlloc(void** ptr, size_t N) {
  CUDA_ASSERT(cudaMalloc(ptr, N));
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void mpMemcpyHtoD(void* dev_ptr, const void* cpu_ptr, size_t bytecount) {
  CUDA_ASSERT(cudaMemcpy(dev_ptr, cpu_ptr, bytecount, cudaMemcpyHostToDevice));
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void mpMemcpyDtoH(void* cpu_ptr, void const* dev_ptr, size_t bytecount) {
  CUDA_ASSERT(cudaMemcpy(cpu_ptr, dev_ptr, bytecount, cudaMemcpyDeviceToHost));
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void mpMemFree(void* dev_ptr) {
  CUDA_ASSERT(cudaFree(dev_ptr));
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void mpDeviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}

