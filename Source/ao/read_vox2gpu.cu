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

#define TILE_SIZE 8

texture<float, cudaTextureType2DLayered> tex; // "dim" filled in the texture reference template is now deprecated

/////////////////////////////////////////////////////////////////////////
// CUDA Kernel
// NOTE: At this point, this is just a test to ensure we are correctly saving
// Per layer: fetch layer's texture data and transform it to write to 3D output array
// CC: work in progress
__global__ void ao_kernel(float *ad_data, float *bd_data, float *c_data, int width, int height, int depth) 
{
  // block ID
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int bz = blockIdx.z;

  // thread ID
  unsigned int idx = threadIdx.x;
  unsigned int idy = threadIdx.y;
  unsigned int idz = threadIdx.z;

  __shared__ float as[1][1][1][3];
  __shared__ float bs[TILE_SIZE][TILE_SIZE][TILE_SIZE][3];

  for (unsigned int m = 0; m < width/TILE_SIZE; m++) {
	as[1][1][1][0] = ad[bx][by][bz][0];
	as[1][1][1][1] = ad[bx][by][bz][1];
	as[1][1][1][2] = ad[bx][by][bz][2];

	bs[idx][idy][idz][0] = bd[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][0];
	bs[idx][idy][idz][1] = bd[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][1];
	bs[idx][idy][idz][2] = bd[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][2];
	
	__syncthreads();

	c_data[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][0] = bs[idx][idy][idz][0] * as[1][1][1][0];
	c_data[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][1] = bs[idx][idy][idz][1] * as[1][1][1][1];
	c_data[idx + bx * blockDim.x][idy + by * blockDim.y][idz + bz * blockDim.z][2] = bs[idx][idy][idz][2] * as[1][1][1][2];

	__syncthreads(); 
  }

  /*
  // index of the first sub-matrix of A, B processed by the block
  int bBegin = height * TILE_SIZE * by;

  // index of the last sub-matrix of A, B processed by the block
  int bEnd = bBegin + height - 1;

  // step size used to iterate through the sub-matrices of A, B
  int bStep = TILE_SIZE;

  // 0.5f offset
  //float u = (x+0.5f) / (float) width;
  //float v = (y+0.5f) / (float) height;
  //int layer = z;

  // loop through all sub-matrices
  for (int i = bBegin; i <= bEnd; i += bStep) {
	__shared__ float bs[TILE_SIZE][TILE_SIZE][TILE_SIZE][3];
	__shared__ float as[bx][by][bz][3];

  	bs[idx][idy][idz][0] = bd_data[idx][idy][idz][0];
	bs[idx][idy][idz][1] = bd_data[idx][idy][idz][1];
	bs[idx][idy][idz][2] = bd_data[idx][idy][idz][2];

	as[bx][by][bz][0] = ad_data[bx][by][bz][0];
	as[bx][by][bz][1] = ad_data[bx][by][bz][1];
	as[bx][by][bz][2] = ad_data[bx][by][bz][2];

	__syncthreads(); 

	for (int k = 0; k < 3; k++) {
	  c_data[idx][idy][idz][k] = bs[bx][by][bz][k] * as[bx][by][bz][k];
	}

	__syncthreads();
  }
  */

  // read from texture, compute occlusion as a function of distance, and write to shared memory
  //float d = sqrtf(pow(dx,2) + pow(dy,2) + pow(dz,2));
  //temp_out[ini_x][ini_y][ini_z][i] = -tex2DLayered(tex, u, v, layer) * d;

  // read from texture, do expected layered transformation and write to global memory
  //g_odata[layer * width * height + y * width + x] = -tex2DLayered(tex, u, v, layer) + layer;

  //int offset = x + y * blockDim.x * gridDim.x;
  //if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

extern "C"
void run_kernel(int width, int height, int depth, int w, int h, int d,
        unsigned int mem_size, unsigned int mem_size_a, unsigned int mem_size_b, float *a, float *b, float *voxel_data, float *voxel_odata)
{
  // allocate device memory for A and B
  float *ad, *bd;
  checkCudaErrors(cudaMalloc((void**) &ad, mem_size_a));
  checkCudaErrors(cudaMalloc((void**) &bd, mem_size_b));

  // copy A and B to allocated device memory locations
  checkCudaErrors(cudaMemcpy(ad, a, mem_size_a, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bd, b, mem_size_b, cudaMemcpyHostToDevice));

  // allocate device memory for temporary result
  float *temp_data = NULL;
  checkCudaErrors(cudaMalloc((void**) &temp_data, mem_size_b));   

  // allocate device memory for device result
  float *c = NULL;
  checkCudaErrors(cudaMalloc((void**) &c, mem_size_a));

  // allocate array and copy data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cu_3darray;
  checkCudaErrors( cudaMalloc3DArray( &cu_3darray, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered ));
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
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 dimGrid(w/TILE_SIZE, h/TILE_SIZE, d/TILE_SIZE);

  /*ao_kernel<<< dimGrid, dimBlock >>>( d_data, width, height, 0); // warmup (for better timing)
    
  // check if kernel execution generated an error
  getLastCudaError("warmup Kernel execution failed"); */

  checkCudaErrors( cudaDeviceSynchronize() );

  sdkCreateTimer( &timer );
  sdkStartTimer( &timer );

  // execute the kernel
  //for (unsigned int layer = 0; layer < num_layers; layer++)
  //  ao__kernel<<< dimGrid, dimBlock, 0 >>>(d_data, width, height, layer); 

  // execute the kernel
  ao_kernel <<< dimGrid, dimBlock >>> (ad, bd, c, w, h, d);

  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors( cudaDeviceSynchronize() );
  sdkStopTimer( &timer );
  printf("Processing time: %.3f msec\n", sdkGetTimerValue( &timer));
  printf("%.2f Mtexlookups/sec\n", (width*height*depth / (sdkGetTimerValue( &timer) / 1000.0f) / 1e6));
  sdkDeleteTimer( &timer );

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(voxel_odata, c, mem_size_a, cudaMemcpyDeviceToHost) );

  checkCudaErrors(cudaFree(temp_data)); checkCudaErrors(cudaFree(c));
  checkCudaErrors(cudaFree(a)); checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFreeArray(cu_3darray));

  return;
}

