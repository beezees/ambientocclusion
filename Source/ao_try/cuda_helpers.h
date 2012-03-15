#ifndef _CUDA_HELPERS_H_
#define _CUDA_HELPERS_H_

#include <cuda_runtime_api.h>

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

void __checkCudaErrors( cudaError err, const char *file, const int line );
void __getLastCudaError( const char *errorMessage, const char *file, const int line );

#endif

