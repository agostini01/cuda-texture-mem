// Code adapted from stack oveflow:
// https://stackoverflow.com/a/38749995

#include <stdio.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>

texture<float, cudaTextureType3D, cudaReadModeElementType> volumeTexIn;
surface<void, 3> volumeTexOut;

__global__ void
surf_write(float *data, cudaExtent volumeSize, float offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
    {
        return;
    }

    float output = data[z * (volumeSize.width * volumeSize.height) + y * (volumeSize.width) + x];

    surf3Dwrite(output + offset, volumeTexOut, x * sizeof(float), y, z);
}

__global__ void
tex_read(float x, float y, float z)
{
    printf("texture: x: %f, y: %f, z:%f, val: %f\n", x, y, z, tex3D(volumeTexIn, x, y, z));
}

__global__ void
regular_read(int x, int y, int z, float * array)
{
    printf("regular: x: %d, y: %d, z:%d, val: %f\n", x, y, z, array[x+y*8+z*64]);
}

void runtest(float *data, cudaExtent vol, float x, float y, float z)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t content;
    checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc, vol, cudaArraySurfaceLoadStore));

    float *d_data;
    checkCudaErrors(cudaMalloc(&d_data, vol.width * vol.height * vol.depth * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, data, vol.width * vol.height * vol.depth * sizeof(float), cudaMemcpyHostToDevice));
    
    regular_read<<<1, 1>>>((int)x, (int)y, (int)z, d_data);

    // Binding the texture and surface
    checkCudaErrors(cudaBindSurfaceToArray(volumeTexOut, content));
    checkCudaErrors(cudaBindTextureToArray(volumeTexIn, content));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((vol.width+7)/8,(vol.height+7)/8, (vol.depth+7)/8);
    surf_write<<<gridSize, blockSize>>>(d_data,vol, 0);

    volumeTexIn.filterMode = cudaFilterModePoint; // Non interpolated
    tex_read<<<1, 1>>>(x, y, z);
    
    volumeTexIn.filterMode = cudaFilterModeLinear; // Interpolated
    tex_read<<<1, 1>>>(x, y, z);
    
    surf_write<<<gridSize, blockSize>>>(d_data,vol, 42);
    
    volumeTexIn.filterMode = cudaFilterModePoint; // Non interpolated
    tex_read<<<1, 1>>>(x, y, z);
    
    volumeTexIn.filterMode = cudaFilterModeLinear; // Interpolated
    tex_read<<<1, 1>>>(x, y, z);
    
    
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFreeArray(content);
    cudaFree(d_data);
    return;
}

int main()
{
    const int dim = 8;
    float *data = (float *)malloc(dim * dim * dim * sizeof(float));
    for (int z = 0; z < dim; z++)
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++)
            {
                data[z * dim * dim + y * dim + x] = z * 100 + y * 10 + x;

                // printf("x: %d, y: %d, z:%d, val: %f\n", 
                //     x, y, z, data[z * dim * dim + y * dim + x]);
            }

    cudaExtent vol = {dim, dim, dim};
    runtest(data, vol, 1.5, 1.5, 1.5);
    runtest(data, vol, 1.9, 1.9, 1.9);
    runtest(data, vol, 5, 5, 5);

    free(data);
    return 0;
}