#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "kernel_header.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

EXTERN_C void mpInitDevice(uint32_t);
EXTERN_C void* mpMemAlloc(len_t N, StreamContext);
EXTERN_C void mpMemcpyHtoD(void* dptr, void const* hptr, len_t N, StreamContext);
EXTERN_C void mpMemcpyDtoH(void* hptr, void const* dptr, len_t N, StreamContext);
EXTERN_C void mpMemFree(void* dptr, StreamContext);
EXTERN_C void mpDeviceSynchronize();
EXTERN_C void mpStreamSynchronize(StreamContext);
EXTERN_C StreamContext mpInitStream();
EXTERN_C void mpDeinitStream(StreamContext);
EXTERN_C void mpCheckLastError();
EXTERN_C len_t mpDeviceTotalMemory(uint32_t);

#endif
