// read_vox2gpu.cu
// Description: Import data to CPU memory, initiate 3D data array for the layered texture 
//
// TODO: check on midas + remove transform kernel - done
// TODO: distribute work accordingly blockDIm(8,8,8) screen pixel per block
// TODO: compute ao
// TODO: add timers - done
// TODO: clean up code

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

#define TILE_SIZE 512

texture<float, cudaTextureType2DLayered> tex; // "dim" filled in the texture reference template is now deprecated

/////////////////////////////////////////////////////////////////////////
// CUDA Kernel
// NOTE: At this point, this is just a test to ensure we are correctly saving
// Per layer: fetch layer's texture data and transform it to write to 3D output array
// CC: work in progress

__global__ void ao_kernel(int *devPtr, int pitch, float *c_data, int size_a) 
{
    int ty = threadIdx.y;
    int by = blockIdx.y;

    float x, y, z;
    float u, v;
    int layer;
    float dx, dy, dz;

    for (int i = 0; i <= size_a/TILE_SIZE ; i += TILE_SIZE) {
      __shared__ float bs[TILE_SIZE];
      unsigned aindex = ty + by * TILE_SIZE;

      int* row = (int*)((char*)devPtr + aindex * pitch);
      x = row[0];
      y = row[1];
      z = row[2];

      u = (x + 0.5f)/(float) 256;
      v = (y + 0.5f)/(float) 256;
      layer = (int) floor(z+0.5);
   
      if( tex2DLayered(tex, (float)u, (float)v, layer) != 0)
     	 bs[ty] = 3; 
   
      /* the output is indexed in the same manner as A therefore there is no need to duplicate x,y,z */
      //c_data[aindex] = temp; /* FOR TESTING ONLY */
    
      __syncthreads(); 
   
      // reduction of all values in bs per block 
      int i = TILE_SIZE/2;
      while (i != 0) { 
	if (ty < i) 
	   bs[ty] += bs[ty+1];
	__syncthreads();
	i /= 2;
      }
   
      // save reduced result in resultant array 
      if (ty == 0) c_data[aindex] = bs[0];
    } 
}

  // read from texture, do expected layered transformation and write to global memory
  //g_odata[layer * width * height + y * width + x] = -tex2DLayered(tex, u, v, layer) + layer;

extern "C"
void run_kernel(int width, int height, int depth, unsigned int size_a, int *a[], float *voxel_data, float *voxel_odata)
{
  // allocate device memory for A 
  int* devPtr;
  size_t pitch;
  checkCudaErrors(cudaMallocPitch((void**)&devPtr, &pitch, 3 * sizeof(int), size_a));

  // copy A to allocated device memory locations
  checkCudaErrors(cudaMemcpy2D(devPtr, pitch, *a, 3*sizeof(int), 3*sizeof(int), size_a, cudaMemcpyHostToDevice));

  // allocate device memory for device result
  float *c = NULL;
  checkCudaErrors(cudaMalloc((void**) &c, size_a * sizeof(float)));

  // allocate array and copy data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cu_3darray;
  checkCudaErrors( cudaMalloc3DArray( &cu_3darray, &channelDesc, make_cudaExtent(width, height, depth), cudaArrayLayered ));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0,0,0);
  myparms.dstPos = make_cudaPos(0,0,0);
  myparms.srcPtr = make_cudaPitchedPtr(voxel_data, width * sizeof(float), width, height);
  myparms.dstArray = cu_3darray;
  myparms.extent = make_cudaExtent(width, height, depth);
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
  
  // setup execution parameters
  dim3 dimBlock(TILE_SIZE,1);
  dim3 dimGrid(size_a/TILE_SIZE,1);

  checkCudaErrors(cudaDeviceSynchronize());

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // execute the kernel
  ao_kernel <<< dimGrid, dimBlock >>> (devPtr, pitch, c, size_a); 

  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer( &timer );
  printf("Processing time: %.3f msec\n", sdkGetTimerValue( &timer));
  printf("%.2f Mtexlookups/sec\n", (width*height*depth / (sdkGetTimerValue( &timer) / 1000.0f) / 1e6));
  sdkDeleteTimer(&timer);

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(voxel_odata, c, size_a*sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(c));
  checkCudaErrors(cudaFree(devPtr)); 
  checkCudaErrors(cudaFreeArray(cu_3darray));

  return;
}

