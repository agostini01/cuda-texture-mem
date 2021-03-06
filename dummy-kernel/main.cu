#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>

#define PI_F 3.141592654f

#define DEBUG 1
//undef DEBUG

void printArrayAsMatrix(const float* in, 
    const size_t& width, const size_t& height) {
#ifdef DEBUG
  std::cout <<"Printing "<<width<<","<<height<<" array"<< std::endl;
  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      std::cout 
        <<std::fixed 
        << std::setw(12) // space between numbers
        << std::setprecision(8) // nubmers after decimal point
        // << std::setprecision(std::numeric_limits<float>::digits10) // nubmers after decimal point
        << in[width*j + i] << ',';
    }
    std::cout << std::endl;
  }
#endif
}

__global__ void rotateKernel (float * output,
    cudaTextureObject_t texObj, int width, int height,
    float theta) {

  // Calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  float u = x / (float)width;
  float v = y / (float)height;
 
  // And regular coordinates
  unsigned int idx= y * width + x;

  // Transform coordinates
  u -= 0.5f;
  v -= 0.5f;
  float tu = u * 1 + 0.5f;
  float tv = v * 1 + 0.5f;

  // Read from texture and write to global memory
  output[idx] = tex2D<float>(texObj, x, y);
}

int main () 
{
  // Inputs
  size_t width = 128;
  size_t height = 128;
  size_t size = width * height * sizeof(float);
  float angle = 0; // in degrees
  float theta = angle/180*PI_F; // in rad

  // Initialize host array 
  float * h_data = (float*)malloc(size);
  for (int i =0; i<height*width; ++i) h_data[i] =(float)i/(height*width);
  memset(h_data, 0, size/4);

  // cudaArray obj will have elements of 32bits, representing single-precision
  // floating point numbers
  cudaChannelFormatDesc ch_desc =
    cudaCreateChannelDesc(32,0,0,0,
        cudaChannelFormatKindFloat);

  cudaArray* cu_array;
  checkCudaErrors(cudaMallocArray(&cu_array, &ch_desc, width, height));

  checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, h_data, size,
      cudaMemcpyHostToDevice));

  // Specify texture
  // Texture is going to be bound to a 1D Array, with name cu_array
  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = cu_array;

  // Specify texture object parameters
  // - Clamp mode: if out of bounds clamp index to closest 0 or width | 0 or height
  // - Without interpoation
  // - No conversion/normalization of the value read
  // - Coordinates are normalized to -1,1: useful for trigonometry
  struct cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0]   = cudaAddressModeClamp;
  tex_desc.addressMode[1]   = cudaAddressModeClamp;
  tex_desc.filterMode       = cudaFilterModePoint;
  tex_desc.readMode         = cudaReadModeElementType;
  tex_desc.normalizedCoords = 0;

  // Copy host memory to cudaArray
  checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, h_data, size,
      cudaMemcpyHostToDevice));

  // Create texture object
  cudaTextureObject_t tex_obj = 0;
  cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
  

  // Allocate result of transformation in device memory
  float* d_output;
  checkCudaErrors(cudaMalloc(&d_output, size));
  
  // Print host array
  printArrayAsMatrix(h_data, width, height);

  // Invoke kernel rotating it once
  dim3 dimBlock(16, 16);
  dim3 dimGrid(
      (width  + dimBlock.x - 1) / dimBlock.x+1,
      (height + dimBlock.y - 1) / dimBlock.y+1);
  rotateKernel<<<dimGrid, dimBlock>>>(d_output,
      tex_obj, width, height,
      theta);
  
  // Print result array
  checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));
  printArrayAsMatrix(h_data, width, height);
  
  // Copy old result to texture and Invoke kernel rotating it again
  checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, d_output, size,
      cudaMemcpyDeviceToDevice));
  rotateKernel<<<dimGrid, dimBlock>>>(d_output,
      tex_obj, width, height,
      theta);
  
  // Print result array
  checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));
  printArrayAsMatrix(h_data, width, height);
  
  // Copy old result to texture and Invoke kernel rotating it again
  checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, d_output, size,
      cudaMemcpyDeviceToDevice));
  rotateKernel<<<dimGrid, dimBlock>>>(d_output,
      tex_obj, width, height,
      theta);
  
  // Print result array
  checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));
  printArrayAsMatrix(h_data, width, height);

  // Destroy texture object
  checkCudaErrors(cudaDestroyTextureObject(tex_obj));

  // Free device memory
  checkCudaErrors(cudaFreeArray(cu_array));
  checkCudaErrors(cudaFree(d_output));

  // Free host memory
  free(h_data);
}
