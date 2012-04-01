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
        //vox_coords << i << " " << j << " " << k << endl;
          vox_coords << j << " " << k << " " << i << endl; 
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

int** read_input_data(const char* ifilename, int *size_a)
{
  int **a =  NULL;
  ifstream *input = new ifstream(ifilename, ios::in);
  char line[256];
  int i = 0;

  // not very nice: read the file in two-passes:
  // first to get the size for memory allocation purposes and
  // second time to read the data

  // first pass:
  if(!input) {
    cerr << "Error: when opening the file " << ifilename << endl;
    exit(1);
  } 
  while(!input->eof()) {
    input->getline(line, 255);
    (*size_a)++;
  }
  *size_a -= 1;

  // rewind at the beginning
  input->clear();
  input->seekg(0, ios::beg);

  // allocate memory for a
  a = (int**) malloc(*size_a * sizeof(int *));
  if(a == NULL) {
    cerr << "Error: out of memory" << endl;
    exit(1);
  }
  for(int i = 0; i < *size_a; i++) {
    a[i] = (int*) malloc(3 * sizeof(int));
    if (a[i] == NULL) {
      cerr << "Error: out of memory" << endl;
      exit(1);
    }
  } 

  // second pass: read data and fill-in a
  while(!input->eof()) {
    if (i == *size_a) // avoid segfault in last pass
      break;
    *input >> a[i][0] >> a[i][1] >> a[i][2];
    i++;
  }
  input->close();

  return a;
}

void save_output_data(const char *ofilename, int **a, int size_a, float *out)
{
  ofstream outfile(ofilename, ios::out);
  for(int i=0; i < size_a; i++) {
    outfile << a[i][0] << " " << a[i][1] << " " << a[i][2] << " " << out[i] << endl;
  }
  outfile.close();
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

  // read voxelized screen pixels from input data from file
  // the size of a is returned in size_a
  // memory is allocated by read_input_data 
  int size_a = 0;
  int **a = read_input_data("filledVox.dat", &size_a);
  cout << "a has " << size_a << " rows" << endl;

#ifdef TESTING /* TEST_ONLY */
  // generate input for voxelized screen pixels // CC
  int size_a = 128*512;
  int **a;

  // allocate memory for a
  a = (int**) malloc(size_a * sizeof(int *));
  if(a == NULL) {
    cerr << "Error: out of memory" << endl;
    exit(1);
  }
  for(int i = 0; i < size_a; i++) {
    a[i] = (int*) malloc(3 * sizeof(int));
    if (a[i] == NULL) {
      cerr << "Error: out of memory" << endl;
      exit(1);
    }
  } 

  // fill in a
  for(int i = 0; i < size_a; i++) {
    a[i][0] = i;
    a[i][1] = i;
    a[i][2] = i;
  }
#endif

  // generate input data for layered texture
  for (unsigned int layer = 0; layer < depth; layer++) {
    for (int i = 0; i < (int)(width * height); i++) {
      voxel_data[layer*width*height + i] = (float)voxels[i];
    }
  }

  // allocate mem for the result on host side // CC
  float *voxel_odata = (float*) malloc(size_a*sizeof(float));
  
  run_kernel(width, height, depth, size_a, a, voxel_data, voxel_odata);

#if 0
  /* TEST for kernel test0 and test1 only */
  for (int i=0; i<size_a; i++) {
    if (fabs(voxel_odata[i] - (a[i][0] + a[i][1] + a[i][2])) > 0.1)
      cerr << "Test failed at " << i << endl;
  }
#endif

#if 1
  /* TEST for kernel test4 only */
  float dx = 0.0f, dy = 0.0f, dz = 0.0f;
  float temp = 0.0f;  
  int dim = 4;
  int pos = 0.0f; int next_pos = 0.0f;
  float diff= 0.0f; int count = 0.0; 
 
  for (unsigned int t=0; t<size_a; t++) {
  pos = a[t][2] * width * height + a[t][1] * width + a[t][0];
    for (int m=-(dim/2); m<=(dim/2)-1; m++) {                    // depth
      for (int n=-(dim/2)+1; n<=(dim/2); n++) {                  // height
        for (int p=-(dim/2); p<=(dim/2)-1; p++) {                // width
              next_pos = (a[t][2] + m) * width * height + (a[t][1] + n) * width + (a[t][0] + p);
              if (voxel_data[pos] != 0 && voxel_data[next_pos] !=0 && (a[t][2] + m)>=0 && (a[t][1] + n)>=0 && (a[t][0] + p)>=0 && (a[t][2] + m)<256 && (a[t][1] + n)<256 && (a[t][0] + p)<256) {
                   dx = powf(p,2);
                   dy = powf(n,2);
                   dz = powf(m,2);
                   temp += dx + dy + dz;
               }
               else temp += 0;
         }
       }
     }
 
   if (fabs(voxel_odata[t] - temp) > 0.1) {
        //cerr << "Distance test failed at " << t << "with position: " << pos << endl;
   	count++; diff += fabs(voxel_odata[t] - temp);
   }     
   temp = 0.0f;

   //cout << "Total difference: " << diff << " for # voxels: " << count << endl;  
 }
 

  /* TEST for kernel test4 with specific values only */
  /*float dx = 0.0f, dy = 0.0f, dz = 0.0f;
  float temp = 0.0f;  
  int dim = 4;
  int const i = 8; //3489

  int pos = a[i][2] * width * height + a[i][1] * width + a[i][0];
  for (int m=-(dim/2); m<=(dim/2)-1; m++) {                    // depth
    for (int n=-(dim/2)+1; n<=(dim/2); n++) {                  // height
      for (int p=-(dim/2); p<=(dim/2)-1; p++) {                // width
                int next_pos = (a[i][2] + m) * width * height + (a[i][1] + n) * width + (a[i][0] + p);
                if (voxel_data[pos] != 0 && voxel_data[next_pos] !=0 && (a[i][2] + m)>=0 && (a[i][1] + n)>=0 && (a[i][0] + p)>=0 && (a[i][2] + m)<256 && (a[i][1] + n)<256 && (a[i][0] + p)<256) {
                  dx = powf(p,2);
                  dy = powf(n,2);
                  dz = powf(m,2);
                  temp += dx + dy + dz;
		  cout << "pos (x,y,z): " << a[i][0] + p << ", " << a[i][1] + n << ", " << a[i][2] + m << endl;
		  cout << "temporary temp value : " << temp << endl;
                }
                else temp += 0;
      }
    }
  }
   
  cout << "diff: " << fabs(voxel_odata[i] - temp) << endl;
  cout << "gpu: " << voxel_odata[i] << " cpu: " << temp << endl;
  cout << "x: " << a[i][0] << " y: " << a[i][1] << " z: " << a[i][2] << endl;
  */

  /* TEST for kernel test3 with 4 neighbors only */
  /* for (int i=0; i<size_a; i++) {
    // correct neighbors?
    int pos = a[i][2] * width * height + a[i][1] * width + a[i][0];
    int pos_top = a[i][2] * width * height + (a[i][1]-1) * width + a[i][0];
    int pos_bottom = a[i][2] * width * height + (a[i][1]+1) * width + a[i][0];
    int pos_left = a[i][2] * width * height + a[i][1] * width + a[i][0]+1;
    int pos_right = a[i][2] * width * height + a[i][1] * width + a[i][0]-1;
    // if (fabs(voxel_odata[i] - voxel_data[pos] - voxel_data[pos_top] - voxel_data[pos_bottom] - voxel_data[pos_left] - voxel_data[pos_right]) > 0.1) 
    //	cerr << "Neighborhood test failed at "<< i << "with position: " << pos << endl;
    // correct summed up weighted distance?
    float temp = voxel_data[pos_top] * powf(a[i][1]-1-a[i][1],2) + voxel_data[pos_bottom] * powf(a[i][1]+1-a[i][1],2) + voxel_data[pos_left] * powf(a[i][0]+1-a[i][0],2) + voxel_data[pos_right] * powf(a[i][0]-1-a[i][0],2);
    if (fabs(voxel_odata[i] - temp) > 0.1)
	cerr << "Distance test failed at " << i << "with position: " << pos << endl;
  } */

  /* TEST for kernel test3 specific values only */
  /*  int const t = 6; 
    int pos = a[t][2] * width * height + a[t][1] * width + a[t][0];
    int pos_top = a[t][2] * width * height + (a[t][1]-1) * width + a[t][0];
    int pos_bottom = a[t][2] * width * height + (a[t][1]+1) * width + a[t][0];
    int pos_left = a[t][2] * width * height + a[t][1] * width + a[t][0] + 1;
    int pos_right = a[t][2] * width * height + a[t][1] * width + a[t][0] - 1;
    cout << fabs(voxel_odata[t] - voxel_data[pos] - voxel_data[pos_top] - voxel_data[pos_bottom] - voxel_data[pos_left]- voxel_data[pos_right]) << endl;
    cout << "gpu: " << voxel_odata[t] << "cpu: " << (float)(voxel_data[pos]+voxel_data[pos_top]+voxel_data[pos_bottom]+voxel_data[pos_right]+voxel_data[pos_left]) << endl;
    cout << "x: " << a[t][0] << "y: " << a[t][1] << "z: " << a[t][2] << endl;
    cout << "next x: " << a[t][0] << "y: " << (a[t][1]-1) << "z: " << a[t][2] << endl;
    cout << "pos: " << pos << ", next pos: " << pos_top << endl;
  */
#endif

  // save the output
  save_output_data("result.dat", a, size_a, voxel_odata);

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
