
#include "device_utils.h"

extern "C" void* mpMemAlloc(size_t N, Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}
extern "C" void mpMemcpyHtoD(void* dev_ptr, const void* cpu_ptr, size_t N, Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}
extern "C" void mpMemcpyDtoH(void* cpu_ptr, void const* dev_ptr, size_t N, Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}
extern "C" void mpMemFree(void* dev_ptr, Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}
extern "C" void mpStreamSynchronize(Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}
extern "C" void mpDeviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" Stream mpInitStream() {
  CUstream stream_ptr = nullptr;
  CURESULT_ASSERT(cuStreamCreate(&stream_ptr, CU_STREAM_DEFAULT));
  return Stream { .ptr = reinterpret_cast<void*>(stream_ptr) };
}
extern "C" void mpDeinitStream(Stream stream) {
  CUstream _stream = static_cast<CUstream>(stream.ptr);
  CURESULT_ASSERT(cuStreamDestroy(_stream));
}
extern "C" void initDevice(unsigned device_number) {

    CURESULT_ASSERT(cuInit(0));

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
