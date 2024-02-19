#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "tensor_types.h"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

typedef unsigned long long ull;

EXTERN_C void mpMemAlloc(void**, len_t);
EXTERN_C void mpMemcpyHtoD(void*, void const*, len_t);
EXTERN_C void mpMemcpyDtoH(void*, void const*, len_t);
EXTERN_C void mpMemFree(void*);
EXTERN_C void mpDeviceSynchronize();

#endif
