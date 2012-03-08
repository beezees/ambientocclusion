// read_vox2gpu.cu
// Description: Import data to CPU memory, initiate 3D data array for the layered texture 
//
// TODO: check on midas + remove transform kernel

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "sdkHelper.h"  // helper for shared that are common to CUDA SDK samples
#include "shrQATest.h"  // This is for automated testing output (--qatest)

#include "cuda_helpers.h"

texture<float, cudaTextureType2DLayered> tex; // "dim" filed in the texture reference template is now deprecated

/////////////////////////////////////////////////////////////////////////
// CUDA Kernel
// NOTE: At this point, this is just a test to ensure we are correctly saving
// Per layer: fetch layer's texture data and transform it to write to 3D output array
// NOTE: Test only! GET RID OF THIS 
__global__ void transform_kernel(float *g_odata, int width, int height, int layer) 
{
  // map from threadIdx/BlockIdx to pixel position
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
   
  // 0.5f offset
  float u = (x+0.5f) / (float) width;
  float v = (y+0.5f) / (float) height;

  // read from texture, do expected layered transformation and write to global memory
  g_odata[layer * width * height + y * width + x] = -tex2DLayered(tex, u, v, layer) + layer;

  //int offset = x + y * blockDim.x * gridDim.x;
  //if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

extern "C"
void run_kernel(int width, int height, unsigned int num_layers,
        unsigned int mem_size, float *voxel_data, float *voxel_odata)
{
  // allocate device memory for result
  float *d_data = NULL;
  checkCudaErrors(cudaMalloc((void**) &d_data, mem_size));   

  // allocate array and copy data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cu_3darray;
  checkCudaErrors( cudaMalloc3DArray( &cu_3darray, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered ));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0,0,0);
  myparms.dstPos = make_cudaPos(0,0,0);
  myparms.srcPtr = make_cudaPitchedPtr(voxel_data, width * sizeof(float), width, height);
  myparms.dstArray = cu_3darray;
  myparms.extent = make_cudaExtent(width, height, num_layers);
  myparms.kind = cudaMemcpyHostToDevice;
  checkCudaErrors( cudaMemcpy3D( &myparms));

  // set texture parameters
  tex.addressMode[0] = cudaAddressModeWrap;
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = true;  // access with normalized texture coordinates

  // bind the array to the texture
  checkCudaErrors( cudaBindTextureToArray(tex, cu_3darray, channelDesc));

  StopWatchInterface * timer;
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  printf("Covering 2D data array of %d x %d: Grid size is %d x %d, each block has 8 x 8 threads\n", width, height, dimGrid.x, dimGrid.y);

  transform_kernel<<< dimGrid, dimBlock >>>( d_data, width, height, 0); // warmup (for better timing)
    
  // check if kernel execution generated an error
  getLastCudaError("warmup Kernel execution failed");

  checkCudaErrors( cudaDeviceSynchronize() );

  sdkCreateTimer( &timer );
  sdkStartTimer( &timer );

  // execute the kernel
  for (unsigned int layer = 0; layer < num_layers; layer++)
    transform_kernel<<< dimGrid, dimBlock, 0 >>>(d_data, width, height, layer);
 
  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors( cudaDeviceSynchronize() );
  sdkStopTimer( &timer );
  printf("Processing time: %.3f msec\n", sdkGetTimerValue( &timer));
  printf("%.2f Mtexlookups/sec\n", (width*height*num_layers / (sdkGetTimerValue( &timer) / 1000.0f) / 1e6));
  sdkDeleteTimer( &timer );

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(voxel_odata, d_data, mem_size, cudaMemcpyDeviceToHost) );

  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFreeArray(cu_3darray));

  return;
}

