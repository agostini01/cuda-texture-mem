#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>

void printArrayAsCharMatrix(const float *in, const size_t &width, const size_t &height)
{
  std::cout << std::endl;
  char buffer[4];
  // int ret;
  for (size_t j = 0; j < height; ++j)
  {
    for (size_t i = 0; i < width; ++i)
    {
      // ret = snprintf(buffer, sizeof buffer, "%f", in[width * j + i]);

      // if (ret < 0) {
      //   return EXIT_FAILURE;
      // }
      // if (ret >= sizeof buffer) {
      // }

      std::cout << buffer[0] << ","
                << buffer[1] << ","
                << buffer[2] << ","
                << buffer[3]
                << ' ';
    }
    std::cout << std::endl;
  }
}

void printArrayAsMatrix(const float *in, const size_t &width, const size_t &height)
{
  std::cout << std::endl;
  for (size_t j = 0; j < height; ++j)
  {
    for (size_t i = 0; i < width; ++i)
    {
      std::cout << std::fixed
                << std::setw(5)         // space between numbers
                << std::setprecision(2) // nubmers after decimal point
                << in[width * j + i] << ' ';
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

  if (x < width && y < height)
  {
    float data;
    // Read from input surface
    surf2Dread(&data, inputSurfObj, x * 4, y);
    if (idx == 100)
      printf("%f\n", data);
    // Write to output surface
    data += 2;
    surf2Dwrite(data, outputSurfObj, x * 4, y);
  }
}

__global__ void printKernel(cudaSurfaceObject_t inputSurfObj,
                            int width, int height, int depth)
{
  // Calculate surface coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int idx = z * width * height + y * width + x;

  if (x < width && y < height && z < depth)
  {
    float data;
    // Read from input surface
    surf3Dread(&data, inputSurfObj, x * sizeof(float), y, z);
    printf("(%d,%d,%d):%d = %f\n", x, y, z, idx, data);
    // Write to output surface
  }
}

__global__ void add70(cudaSurfaceObject_t inputSurfObj,
                      int width, int height, int depth)
{
  // Calculate surface coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int idx = z * width * height + y * width + x;

  if (x < width && y < height && z < depth)
  {
    float data;
    // Read from input surface
    surf3Dread(&data, inputSurfObj, x * sizeof(float), y, z);
    printf("(%d,%d,%d):%d = %f\n", x, y, z, idx, data);
    // Write to output surface
    surf3Dwrite(data + 70, inputSurfObj, x * sizeof(float), y, z);
  }
}

int main()
{
  // Inputs
  size_t width = 10;
  size_t height = 10;
  size_t depth = 10;
  size_t size = width * height * depth * sizeof(float);

  // Initialize host array
  float *h_data = (float *)malloc(size);
  for (int z = 0; z < depth; ++z)
    for (int y = 0; y < width; ++y)
      for (int x = 0; x < height; ++x)
        h_data[width * height * z + height * y + x] = (float)x * 1 + (float)y * 1000 + (float)z * 1000000;

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0,
                            cudaChannelFormatKindFloat);

  cudaExtent extent = make_cudaExtent(width, height, depth);

  cudaArray *cuInputArray; //may have to use cuArray3DCreate, the descriptor is automatically generated
  cudaMalloc3DArray(&cuInputArray, &channelDesc, extent,
                    cudaArraySurfaceLoadStore);

  cudaArray *cuOutputArray;
  cudaMalloc3DArray(&cuOutputArray, &channelDesc, extent,
                    cudaArraySurfaceLoadStore);

  // checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, h_data, size,
  //     cudaMemcpyHostToDevice));

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpy3DParms memcpyparmsHtoD = {0};
  // memcpyparmsHtoD.srcPtr = h_data;
  memcpyparmsHtoD.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), height, depth);
  memcpyparmsHtoD.dstArray = cuInputArray;
  memcpyparmsHtoD.extent = extent;
  // memcpyparmsHtoD.extent = make_cudaExtent(width * sizeof(float), height, depth);
  memcpyparmsHtoD.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&memcpyparmsHtoD);
  // cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
  //                   cudaMemcpyHostToDevice);

  // Create the surface objects
  // Specify surface
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuInputArray;

  cudaSurfaceObject_t inputSurfObj = 0;
  cudaCreateSurfaceObject(&inputSurfObj, &resDesc);

  // Invoke kernel
  dim3 dimBlock(4, 4, 4);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  // Copy from original surface and add 70
  std::cout << "Printing" << std::endl;
  printKernel<<<dimGrid, dimBlock>>>(inputSurfObj, width, height, depth);
  std::cout << "Adding..." << std::endl;
  add70<<<dimGrid, dimBlock>>>(inputSurfObj, width, height, depth);
  std::cout << "Printing" << std::endl;
  printKernel<<<dimGrid, dimBlock>>>(inputSurfObj, width, height, depth);

  // Allocate output buffer in device memory
  float *d_output;
  checkCudaErrors(cudaMalloc(&d_output, size));

  // Copy device to device
  cudaMemcpy3DParms memcpyparmsDtoD = {0};
  // memcpyparmsHtoD.srcPtr = h_data;
  memcpyparmsDtoD.dstPtr = make_cudaPitchedPtr(d_output, width * sizeof(float), height, depth);
  memcpyparmsDtoD.srcArray = cuInputArray;
  memcpyparmsDtoD.extent = extent;
  // memcpyparmsHtoD.extent = make_cudaExtent(width * sizeof(float), height, depth);
  memcpyparmsDtoD.kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(&memcpyparmsDtoD);

  checkCudaErrors(cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost));

  // Print new host data
  for (int z = 0; z < depth; ++z)
  {
    std::cout << std::endl;
    for (int y = 0; y < width; ++y)
    {
      std::cout << std::endl;
      for (int x = 0; x < height; ++x)
      {
        std::cout << std::fixed << std::setw(10) << std::setprecision(1)
                  << h_data[width * height * z + height * y + x] << ", ";
      }
    }
  }
  std::cout << std::endl;

  // Destroy surface objects
  cudaDestroySurfaceObject(inputSurfObj);

  // Free device memory
  checkCudaErrors(cudaFreeArray(cuInputArray));
  checkCudaErrors(cudaFreeArray(cuOutputArray));
  checkCudaErrors(cudaFree(d_output));

  // Free other
  free(h_data);

  return 0;
}
