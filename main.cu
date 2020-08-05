#include <stdio.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>

texture<float, cudaTextureType3D, cudaReadModeElementType>
    volumeTexIn;
surface<void, 3> volumeTexOut;

__global__ void
surf_write(float * data, cudaExtent volumeSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
    {
        return;
    }

    float output = data[
        z*(volumeSize.width*volumeSize.height)+y*(volumeSize.width) + x
    ];

    surf3Dwrite(output, volumeTexOut, x * sizeof(float), y, z);
}

__global__ void
tex_read(float x, float y, float z)
{
    printf("x: %f, y: %f, z:%f, val: %f\n", x, y, z, tex3D(volumeTexIn, x, y, z));
}

int main(){}