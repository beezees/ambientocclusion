#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cuda_runtime_api.h>
#include <cutil_math.h>

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include "sdkHelper.h"  // helper for shared that are common to CUDA SDK samples
#include "shrQATest.h"  // This is for automated testing output (--qatest)

void __checkCudaErrors( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
        file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
  }
}

void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
	fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
  }
}

