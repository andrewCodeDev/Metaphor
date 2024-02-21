#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "kernel_header.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

typedef struct {
    void* ptr;
} Stream;

EXTERN_C void initDevice(unsigned int);
EXTERN_C void* mpMemAlloc(len_t N, Stream);
EXTERN_C void mpMemcpyHtoD(void* dptr, void const* hptr, len_t N, Stream);
EXTERN_C void mpMemcpyDtoH(void* hptr, void const* dptr, len_t N, Stream);
EXTERN_C void mpMemFree(void* dptr, Stream);
EXTERN_C void mpDeviceSynchronize();
EXTERN_C void mpStreamSynchronize(Stream);
EXTERN_C Stream mpInitStream();
EXTERN_C void mpDeinitStream(Stream);

#endif
