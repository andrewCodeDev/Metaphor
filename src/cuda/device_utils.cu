
#include "device_utils.h"

extern "C" void* mpMemAlloc(len_t N, StreamContext stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}
extern "C" void mpMemcpyHtoD(void* dev_ptr, const void* cpu_ptr, len_t N, StreamContext stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}
extern "C" void mpMemcpyDtoH(void* cpu_ptr, void const* dev_ptr, len_t N, StreamContext stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}
extern "C" void mpMemFree(void* dev_ptr, StreamContext stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}
extern "C" void mpStreamSynchronize(StreamContext stream) {
  CUstream _stream = get_stream(stream);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}
extern "C" void mpDeviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void mpInitDevice(uint32_t device_number) {

    CURESULT_ASSERT(cuInit(device_number));

    CUdevice device;
    CUcontext context;
    int device_count = 0;

    CURESULT_ASSERT(cuDeviceGetCount(&device_count));

    if (device_count <= device_number) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    CURESULT_ASSERT(cuDeviceGet(&device, device_number));

    CURESULT_ASSERT(cuCtxCreate(&context, 0, device));
}

// Convenience wrapper for cudaGetLastError.
// TODO: make this return values instead of void
extern "C" void mpCheckLastError()
{
  CUDA_ASSERT(cudaDeviceSynchronize());
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error %s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
  }
}

extern "C" len_t mpDeviceTotalMemory(uint32_t device) {
  len_t total;
  CURESULT_ASSERT(cuDeviceTotalMem(&total, device));
  return total;
}

extern "C" StreamContext mpInitStream() {
  cudaStream_t cuda_stream = nullptr;
  cublasHandle_t blas_handle = nullptr;

  // TODO: Add device parameter? This can set devices for creating
  //       the streams before initializing them.
  //          ex: cudaSetDevice()

  CURESULT_ASSERT(cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT));

  CUBLAS_ASSERT(cublasCreate(&blas_handle));

  CUBLAS_ASSERT(cublasSetStream(blas_handle, cuda_stream));

  return { 
    .cuda_stream = { .ptr = reinterpret_cast<void*>(cuda_stream) },
    .blas_handle = { .ptr = reinterpret_cast<void*>(blas_handle) }
  };
}

extern "C" void mpDeinitStream(StreamContext stream) {

  // TODO: If devices get set, it's probably a good idea to capture
  //       which device a stream was created on and put that in the
  //       StreamContext object. Research if it's required to deinit
  //       streams on the correct device.
  
  CUBLAS_ASSERT(cublasDestroy(get_handle(stream)));
  CURESULT_ASSERT(cuStreamDestroy(get_stream(stream)));
}
