// read_vox2gpu.cu
// Description: Import data to CPU memory, initiate 3D data array for the layered texture 
//
// TODO: check on midas + remove transform kernel

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
#include <shrQATest.h>  // This is for automated testing output (--qatest)

#include "read_binvox.cc"

static char *sSDKname = "AOX";

#define W 256
#define H 256
#define D 256

//////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Helper Functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
        file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
	fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
  }
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
	fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
  }
  if (devID < 0) devID = 0;
  if (devID > deviceCount-1) {
	fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
  }

  cudaDeviceProp deviceProp;
  checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
  if (deviceProp.major < 1) {
	fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  \
  }

  checkCudaErrors( cudaSetDevice(devID) );
  printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
  int current_device   = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device  = 0;
  int device_count     = 0, best_SM_arch     = 0;
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount( &device_count );
  // Find the best major SM Architecture GPU device
  while ( current_device < device_count ) {
	cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999) {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
   }

   // Find the best CUDA capable GPU device
   current_device = 0;
   while( current_device < device_count ) {
     cudaGetDeviceProperties( &deviceProp, current_device );
     if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
	  sm_per_multiproc = 1;
     } else {
          sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
     }

     int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
     if( compute_perf  > max_compute_perf ) {
   	  // If we find GPU with SM major > 2, search only these
          if ( best_SM_arch > 2 ) {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch) {
                     max_compute_perf  = compute_perf;
                     max_perf_device   = current_device;
                 }
           } else {
                 max_compute_perf  = compute_perf;
                 max_perf_device   = current_device;
           }
      }
       ++current_device;
    }
    return max_perf_device;
}

// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv)
{
  cudaDeviceProp deviceProp;
  int devID = 0;
  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
  	devID = getCmdLineArgumentInt(argc, argv, "device=");
        if (devID < 0) {
           printf("Invalid command line parameters\n");
           exit(-1);
        } else {
           devID = gpuDeviceInit(devID);
           if (devID < 0) {
               printf("exiting...\n");
               shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
               exit(-1);
            }
        }
   } else {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
   }
   return devID;
}
// end of CUDA Helper Functions

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

/////////////////////////////////////////////////////////////////////////
// Main
int main(int argc, char** argv) 
{
  shrQAStart(argc, argv);

  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  int devID = findCudaDevice((const int)argc, (const char **)argv);
  bool bResult = true;

  // get # SMs on GPU
  cudaDeviceProp deviceProps;

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount );
  printf("SM %d.%d\n", deviceProps.major, deviceProps.minor );

  if (deviceProps.major < 2) {
   	printf("%s requires SM >= 2.0 to support Texture Arrays.  Test will exit... \n", sSDKname);
        cudaDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
   }

   // allocate host memory for voxel data
   unsigned int width = W, height = H, num_layers = D;
   unsigned int mem_size = width * height * num_layers * sizeof(float);
   float *voxel_data = (float*) malloc(mem_size);

   // generate input data for layered texture
   for (unsigned int layer = 0; layer < num_layers; layer++) {
   	for (int i = 0; i < (int)(width * height); i++) {
	  voxel_data[layer*width*height + i] = (float)voxels[i];
        }
   }

   // this is the expected transformation of the input data (the expected output)
   float *voxel_data_ref = (float*) malloc(size);
    for (unsigned int layer = 0; layer < num_layers; layer++)
        for (int i = 0; i < (int)(width * height); i++)
            voxel_data_ref[layer*width*height + i] = -voxel_data[layer*width*height + i] + layer;

   // allocate device memory for result
   float *d_data = NULL;
   checkCudaErrors(cudaMalloc((void**) &d_data, size));   

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
    checkCudaErrors( cudaBindTextureToArray( tex, cu_3darray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    printf("Covering 2D data array of %d x %d: Grid size is %d x %d, each block has 8 x 8 threads\n", width, height, dimGrid.x, dimGrid.y);

    transform_kernel<<< dimGrid, dimBlock >>>( d_data, width, height, 0); // warmup (for better timing)
    
    // check if kernel execution generated an error
    getLastCudaError("warmup Kernel execution failed");

    checkCudaErrors( cudaDeviceSynchronize() );

    StopWatchInterface * timer;
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

    // allocate mem for the result on host side
    float* voxel_odata = (float*) malloc( size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(voxel_odata, d_data, size, cudaMemcpyDeviceToHost) );

    // write regression file if necessary
    if( checkCmdLineFlag( argc, (const char **)argv, "regression") ) {
        // write file for regression test
        sdkWriteFile<float>( "./data/regression.dat", voxel_odata, width*height, 0.0f, false);
    }
    else
    {
        printf("Comparing kernel output to expected data\n");
        #define MIN_EPSILON_ERROR 5e-3f
        bResult = compareData(voxel_odata, voxel_data_ref, width * height * num_layers, MIN_EPSILON_ERROR, 0.0f);
    }

    // cleanup memory
    free(voxel_data);
    free(voxel_data_ref);
    free(voxel_odata);

    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFreeArray(cu_3darray));

    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (bResult ? QA_PASSED : QA_FAILED) );
}
