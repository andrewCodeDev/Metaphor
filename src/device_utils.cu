
//#include "device_utils.h"

#include "kernel_header.h"

extern "C" void* mpMemAlloc(size_t N, void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}
extern "C" void mpMemcpyHtoD(void* dev_ptr, const void* cpu_ptr, size_t N, void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}
extern "C" void mpMemcpyDtoH(void* cpu_ptr, void const* dev_ptr, size_t N, void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}
extern "C" void mpMemFree(void* dev_ptr, void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}
extern "C" void mpStreamSynchronize(void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}
extern "C" void mpDeviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}
extern "C" void* mpInitStream() {
  CUstream stream = nullptr;
  CURESULT_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  return reinterpret_cast<void*>(stream);
}
extern "C" void mpDeinitStream(void* stream) {
  CUstream _stream = static_cast<CUstream>(stream);
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
