#ifndef _READ_VOX2GPU_H_
#define _READ_VOX2GPU_H_

#include <cuda_runtime_api.h>

extern "C" {
  void run_kernel(int width, int height, unsigned int num_layers,
        unsigned int mem_size, float *voxel_data, float *voxel_odata);
};

#endif

