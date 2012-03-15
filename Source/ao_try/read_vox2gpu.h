#ifndef _READ_VOX2GPU_H_
#define _READ_VOX2GPU_H_

#include <cuda_runtime_api.h>

extern "C" {
  void run_kernel(int width, int height, int depth, unsigned int size_a, 
                int *a[], float *voxel_data, float *voxel_odata);
};

#endif

