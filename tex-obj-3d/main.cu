#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>

void printArrayAsCharMatrix(const float* in, const size_t& width, const size_t& height) {
    std::cout << std::endl;
    char buffer[4];
    int ret;
    for (size_t j = 0; j < height; ++j)
    {
      for (size_t i = 0; i < width; ++i)
      {
        ret=snprintf(buffer, sizeof buffer, "%f", in[width*j + i]);

        // if (ret < 0) {
        //   return EXIT_FAILURE;
        // }
        // if (ret >= sizeof buffer) {
        // }
        
        std::cout << buffer[0]
                  << buffer[1]
                  << buffer[2]
                  << buffer[3]
                  << ' ';
      }
      std::cout << std::endl;
  }
}

void printArrayAsMatrix(const float* in, const size_t& width, const size_t& height) {
    std::cout << std::endl;
  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      std::cout <<std::fixed 
        << std::setw(5) // space between numbers
        << std::setprecision(2) // nubmers after decimal point
        << in[width*j + i] << ' ';
    }
    std::cout << std::endl;
  }
}

__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfObj, x * 4, y);
        if(idx==100) printf("%d\n", data.w);
        // Write to output surface
        data.w+=5;
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

int main () 
{
  // Inputs
  size_t width = 16;
  size_t height = 16;
  size_t size = width * height * sizeof(float);

  // Initialize host array 
  float * h_data = (float*)malloc(size);
  for (int i =0; i<height*width; ++i) h_data[i] =(float)i/(height*width);

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 8, 8, 8,
                            cudaChannelFormatKindUnsigned);

  cudaArray *cuInputArray;
  cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                  cudaArraySurfaceLoadStore);

  cudaArray *cuOutputArray;
  cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                  cudaArraySurfaceLoadStore);

  // checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, h_data, size,
  //     cudaMemcpyHostToDevice));


  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                    cudaMemcpyHostToDevice);

  // Specify surface
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

  // Create the surface objects
  resDesc.res.array.array = cuInputArray;
  cudaSurfaceObject_t inputSurfObj = 0;
  cudaCreateSurfaceObject(&inputSurfObj, &resDesc);

  resDesc.res.array.array = cuOutputArray;
  cudaSurfaceObject_t outputSurfObj = 0;
  cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

  // Allocate output buffer in device memory
  float* d_output;
  checkCudaErrors(cudaMalloc(&d_output, size));
  // Print result array
  // checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, d_output, size,
  //     cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpyFromArray(d_output, cuInputArray, 0, 0, size,
  //     cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyFromArray(h_data, cuInputArray, 0, 0, size,
      cudaMemcpyDeviceToHost));
  // printArrayAsMatrix(h_data, width, height);
  printArrayAsCharMatrix(h_data, width, height);


  // Invoke kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
      (height + dimBlock.y - 1) / dimBlock.y);
  copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj,
                                    outputSurfObj,
                                    width, height);
  
  // checkCudaErrors(cudaMemcpyFromArray(d_output, cuOutputArray, 0, 0, size,
  //     cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyFromArray(h_data, cuOutputArray, 0, 0, size,
      cudaMemcpyDeviceToHost));
  // printArrayAsMatrix(h_data, width, height);
  printArrayAsCharMatrix(h_data, width, height);

  // Destroy surface objects
  cudaDestroySurfaceObject(inputSurfObj);
  cudaDestroySurfaceObject(outputSurfObj);

  // Free device memory
  checkCudaErrors(cudaFreeArray(cuInputArray));
  checkCudaErrors(cudaFreeArray(cuOutputArray));
  checkCudaErrors(cudaFree(d_output));

  // Free other
  free(h_data);

  return 0;
}

