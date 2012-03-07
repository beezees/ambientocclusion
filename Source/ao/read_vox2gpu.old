// read_vox2gpu.cu
// Description: First generates a 3D input data array for the layered texture and the expected output. Then it starts CUDA C kernels, one for each layer, which fetch their layer's texture data (using normalized texture coordinates) & transform it to the expected output. and write it to a 3D output data array.

#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 256
#define PI 3.1415926535897932f

// globals needed by the update routine
struc DataBlock {
  unsigned char *output_bitmap;
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;
  CPUAnimBitmap *bitmap;
  cudaEvent_t start, stop;
  float totalTime;
  float Frames;
}

