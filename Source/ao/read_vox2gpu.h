#ifndef _READ_VOX2GPU_H_
#define _READ_VOX2GPU_H_

#include <cuda_runtime_api.h>

extern "C" {
  void run_kernel(int width, int height, int depth, int w, int h, int d, unsigned int mem_size, unsigned int mem_size_a, unsigned int mem_size_b, float *a, float *b, float *voxel_data, float *voxel_odata);
};

#endif

