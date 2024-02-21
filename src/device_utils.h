#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "kernel_header.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

EXTERN_C void initDevice(unsigned int);
EXTERN_C void* mpMemAlloc(len_t N, void* stream);
EXTERN_C void mpMemcpyHtoD(void* dptr, void const* hptr, len_t N, void* stream);
EXTERN_C void mpMemcpyDtoH(void* hptr, void const* dptr, len_t N, void* stream);
EXTERN_C void mpMemFree(void* dptr, void* stream);
EXTERN_C void mpDeviceSynchronize();
EXTERN_C void mpStreamSynchronize(void* stream);
EXTERN_C void* mpInitStream();
EXTERN_C void mpDeinitStream(void* stream);

#endif
