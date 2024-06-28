
#include "device_utils.h"

extern "C" void* mpMemAlloc(len_t N, StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}
extern "C" void mpMemcpyHtoD(void* dev_ptr, const void* cpu_ptr, len_t N, StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}
extern "C" void mpMemcpyDtoH(void* cpu_ptr, void const* dev_ptr, len_t N, StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}
extern "C" void mpMemFree(void* dev_ptr, StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}
extern "C" void mpStreamSynchronize(StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}
extern "C" void mpDeviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" StreamCtx mpInitStream() {
  CUstream stream_ptr = nullptr;
  CURESULT_ASSERT(cuStreamCreate(&stream_ptr, CU_STREAM_DEFAULT));
  return { .ptr = reinterpret_cast<void*>(stream_ptr) };
}
extern "C" void mpDeinitStream(StreamCtx stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CURESULT_ASSERT(cuStreamDestroy(_stream));
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
