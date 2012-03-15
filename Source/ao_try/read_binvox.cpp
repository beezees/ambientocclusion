// read_binvox.cc
// Description: This example program reads a .binvox file and writes an ASCII version of the same file called "voxels.txt"
// 0 = empty voxel
// 1 = filled voxel;
// The x-axis is the most significant axis, then the z-axis, then the y-axis.


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

#include "read_vox2gpu.h"
#include "cuda_helpers.h"

#define W 256
#define H 256
#define D 256

typedef unsigned char byte;

using namespace std;

static int version;
static int depth, height, width;
static int size;
static float tx, ty, tz;
static float scale;
static char *sSDKname = "AOX";
byte *voxels = 0;

//////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Helper Functions

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

int read_binvox(string filespec)
{

  ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);

  //
  // read header
  //
  string line;
  *input >> line;  // #binvox
  if (line.compare("#binvox") != 0) {
    cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
    delete input;
    return 0;
  }
  *input >> version;
  cout << "reading binvox version " << version << endl;

  depth = -1;
  int done = 0;
  while(input->good() && !done) {
    *input >> line;
    if (line.compare("data") == 0) done = 1;
    else if (line.compare("dim") == 0) {
      *input >> depth >> height >> width;
    }
    else if (line.compare("translate") == 0) {
      *input >> tx >> ty >> tz;
    }
    else if (line.compare("scale") == 0) {
      *input >> scale;
    }
    else {
      cout << "  unrecognized keyword [" << line << "], skipping" << endl;
      char c;
      do {  // skip until end of line
        c = input->get();
      } while(input->good() && (c != '\n'));

    }
  }
  if (!done) {
    cout << "  error reading header" << endl;
    return 0;
  }
  if (depth == -1) {
    cout << "  missing dimensions in header" << endl;
    return 0;
  }

  size = width * height * depth;
  voxels = new byte[size];
  if (!voxels) {
    cout << "  error allocating memory" << endl;
    return 0;
  }

  //
  // read voxel data
  //
  byte value;
  byte count;
  int index = 0;
  int end_index = 0;
  int nr_voxels = 0;
  
  input->unsetf(ios::skipws);  // need to read every byte now (!)
  *input >> value;  // read the linefeed char

  while((end_index < size) && input->good()) {
    *input >> value >> count;

    if (input->good()) {
      end_index = index + count;
      if (end_index > size) return 0;
      for(int i=index; i < end_index; i++) voxels[i] = value;
      
      if (value) nr_voxels += count;
      index = end_index;
    }  // if file still ok
    
  }  // while

  input->close();
  cout << "  read " << nr_voxels << " voxels" << endl;

  // 
  // voxel coords
  // 
  ofstream vox_coords("vox_coords.dat", ios::out);
  for(int i=0; i < depth; i++) {
    for(int k=0; k < height; k++) {
       for(int j=0; j < width; j++) {
         int index = i * width * height + k * width + j;
         if (voxels[index]) {
        vox_coords << i << " " << j << " " << k << endl;
         }
       }
     }
  }
  vox_coords.close();

  return 1;
}

void create_data()
{  
  //
  // now write the data to as ASCII
  //
  ofstream *out = new ofstream("voxels.txt");
  if(!out->good()) {
    cout << "Error opening [voxels.txt]" << endl << endl;
    exit(1);
  }

  cout << "Writing voxel data to ASCII file..." << endl;
  
  *out << "#binvox ASCII data" << endl;
  *out << "dim " << depth << " " << height << " " << width << endl;
  *out << "translate " << tx << " " << ty << " " << tz << endl;
  *out << "scale " << scale << endl;
  *out << "data" << endl;

  for(int i=0; i < size; i++) {
    *out << (char) (voxels[i] + '0') << " ";
    if (((i + 1) % width) == 0) *out << endl;
  }

  out->close();

  cout << "done" << endl << endl;
}

int main(int argc, char** argv) 
{
  //argv[1] = "bunny.binvox";
  if (argc != 2) {
    cerr << "Requires model filename [bunny.binvox] as argument!" << endl;
    exit(1);
  }

  if (!read_binvox(argv[1])) {
    cerr << "Error reading [" << argv[1] << "]" << endl << endl;
    exit(1);
  }
  shrQAStart(argc, argv);

  create_data(); 

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
  unsigned int width = W, height = H, depth = D;
  unsigned int mem_size = width * height * depth * sizeof(float);
  float *voxel_data = (float*) malloc(mem_size);

  // allocate host memory for voxelized screen pixels // CC
  int size_a = 128*512;
  int **a;
  a = (int**) malloc(size_a * sizeof(int *));
  if(a == NULL) {
    cerr << "out of memory" << endl;
    exit(1);
  }
  for(int i = 0; i < size_a; i++) {
    a[i] = (int*) malloc(3 * sizeof(int));
    if(a[i] == NULL) {
      cerr << "out of memory" << endl;
      exit(1);
    }
  } 

/* TEST_ONLY */
  // generate input for voxelized screen pixels // CC
  for(int i = 0; i < size_a; i++) {
	a[i][0] = 1;
    	a[i][1] = 1;
	a[i][2] = 1;
  }
/* */

  // generate input data for layered texture
  for (unsigned int layer = 0; layer < depth; layer++) {
    for (int i = 0; i < (int)(width * height); i++) {
      voxel_data[layer*width*height + i] = (float)voxels[i];
    }
  }

  /*// generate input for neighborhood matrix // CC
  int b_i = -1, b_j = -1, b_k = -1;
  int incr = -4;
  for (unsigned int idz = 0; idz < 8 * d; idz++) {
    if (idz % 8 == 0) { b_i++; incr = -4; }
    for (unsigned int idy = 0; idy < 8 * h; idy++) {
       if (idy % 8 == 0) { b_j++; incr = -4; }
       for (unsigned int idx = 0; idx < 8 * w; idx++) {
        if (idx % 8 == 0) { b_k++; incr = -4; }    
        b[idx][idy][idz][0] = a[b_i][b_j][b_k][0] + 0.5f - incr;
        b[idx][idy][idz][1] = a[b_i][b_j][b_k][1] + 0.5f - incr;
        b[idx][idy][idz][2] = a[b_i][b_j][b_k][2] + 0.5f - incr; 
        incr++;
       }
    }
  }*/

  // allocate mem for the result on host side // CC
  float *voxel_odata = (float*) malloc(size_a*sizeof(float));

  run_kernel(width, height, depth, size_a, a, voxel_data, voxel_odata);

  /* TESTING ONLY */
  for (int i=0; i<size_a; i++) {
     if (fabs(voxel_odata[i] - 3*512.0f > 0.1) )
		cerr << "Test failed at " << i << endl;
  }

  // cleanup memory
  free(voxel_data);
  for(int i = 0; i < size_a; i++)
    free(a[i]);
  free(a);
  free(voxel_odata);

  cudaDeviceReset();
  shrQAFinishExit(argc, (const char **)argv, (bResult ? QA_PASSED : QA_FAILED) );
  //shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}
