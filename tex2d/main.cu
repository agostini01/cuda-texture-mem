// adapted from:
// The docs

#include <stdio.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <cuda_fp16.h>

__global__ void transformKernel(float *output, cudaTextureObject_t texObj,
                                int width, int height, float theta)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;

    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // Reads from texture, writes to array
    output[y*width + x] = tex2D<float>(texObj, tu, tv);
}


__global__ void
regular_read(int x, int y, float * array, int width, int height)
{
    printf("regular: x: %d, y: %d, val: %f\n", x, y, array[x+y*width]);
}

int main()
{
    const int dim = 128;
    size_t width = dim;
    size_t height = dim;
    float angle = dim;

    // Initialize Host memory
    float *h_data = (float *)malloc(dim * dim * dim * sizeof(float));
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++)
            h_data[y * dim + x] = y * 10 + x;

    // Initialize a 2D array device array and copy host data to device
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray * cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width*sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Specify the texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocates a regular array on the device
    float * output;
    cudaMalloc(&output, width * height * sizeof(float));
    cudaMemcpy(output,h_data,width*height*sizeof(float), cudaMemcpyHostToDevice);


    // angle = 0;
    // Grid/block dims to transform the array
    dim3 dimBlock(16,16);
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y);
    
    regular_read<<<1, 1>>>(20,20, output, width, height);
    transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, angle);
    regular_read<<<1, 1>>>(20,20, output, width, height);


    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(output);
    free(h_data);
    return 0;



}