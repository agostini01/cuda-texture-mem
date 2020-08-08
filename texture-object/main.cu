#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include <numeric>
#include <vector>
#include <iostream>


__global__ void transformKernel (float * output,
    cudaTextureObject_t texObj,
    int width, int height,
    float theta)
{
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
  float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
  float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
  // Read from texture and write to global memory
  output[idx] = tex2D<float>(texObj, tu, tv);
  
  if(idx==10) printf("%f\n", output[idx]);
  if(idx==10) printf("%f\n", tex2D<float>(texObj, tu, tv));

}



int main () 
{
  // Inputs
  size_t width = 128;
  size_t height = 128;
  size_t size = width * height * sizeof(float);
  float angle = 0;

  // Initialize host array 
  std::vector<int> h_data(width*height);
  std::iota(h_data.begin(), h_data.end(), 0); // vector with increasing integers
  

  // cudaArray obj will have elements of 32bits, representing single-precision
  // floating point numbers
  cudaChannelFormatDesc ch_desc =
    cudaCreateChannelDesc(32,0,0,0,
        cudaChannelFormatKindFloat);

  cudaArray* cu_array;
  checkCudaErrors(cudaMallocArray(&cu_array, &ch_desc, width, height));

  checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, h_data.data(), size,
      cudaMemcpyHostToDevice));

  // Specify texture
  // Texture is going to be bound to a 1D Array, with name cu_array
  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = cu_array;

  // Specify texture object parameters
  // - Wrap mode: when outside of the boorderd, index x is converted to
  //   frac(x)=x floor(x) with floor(x) is the largest integer 
  //   not greater than x
  // - With interpoation
  // - No conversion/normalization of the value read
  // - Coordinates are normalized to -1,1: useful for trigonometry
  struct cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0]   = cudaAddressModeWrap;
  tex_desc.addressMode[1]   = cudaAddressModeWrap;
  tex_desc.filterMode       = cudaFilterModeLinear;
  tex_desc.readMode         = cudaReadModeElementType;
  tex_desc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t tex_obj = 0;
  cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

  // Allocate result of transformation in device memory
  float* output;
  checkCudaErrors(cudaMalloc(&output, size));
  
  // Print input and output elements
  for (int i =10; i<15; ++i)
    std::cout << "("<<i<<","<<h_data[i]<<") ";
  std::cout<<std::endl;

  // Invoke kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
      (height + dimBlock.y - 1) / dimBlock.y);
  transformKernel<<<dimGrid, dimBlock>>>(output,
      tex_obj, width, height,
      angle);

  checkCudaErrors(cudaMemcpy(h_data.data(), output, size, cudaMemcpyDeviceToHost));


  // Print input and output elements
  for (int i =10; i<15; ++i)
    std::cout << "("<<i<<","<<h_data[i]<<") ";
  std::cout<<std::endl;


  // Destroy texture object
  checkCudaErrors(cudaDestroyTextureObject(tex_obj));

  // Free device memory
  checkCudaErrors(cudaFreeArray(cu_array));
  checkCudaErrors(cudaFree(output));
}
