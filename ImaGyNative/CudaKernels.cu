#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(err_code) do { 
    cudaError_t _err = (err_code); 
    if (_err != cudaSuccess) { 
        /* In a real app, you\'d want to log this error. */ 
        /* For now, we just return false to signal failure. */ 
        goto error_exit; 
    } 
} while (0)

// ---------------- Pixel-wise Operations ----------------
__global__ void BinarizationKernel(unsigned char* data, int width, int height, int stride, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * stride + x;
    data[idx] = (data[idx] > threshold) ? 255 : 0;
}

bool LaunchBinarizationKernel(unsigned char* pixels, int width, int height, int stride, int threshold)
{
    unsigned char* d_pixels = nullptr;
    size_t imageSize = (size_t)height * stride;

    // 1. Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_pixels, imageSize));

    // 2. Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_pixels, pixels, imageSize, cudaMemcpyHostToDevice));

    // 3. Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 4. Launch the kernel
    BinarizationKernel<<<grid, block>>>(d_pixels, width, height, stride, (unsigned char)threshold);
    
    // Check for kernel launch errors
    cudaError_t lastError = cudaGetLastError();
    CUDA_CHECK(lastError);

    // 5. Wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy data from device back to host
    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, imageSize, cudaMemcpyDeviceToHost));

    // 7. Free device memory
    // Clean up resources
    if (d_pixels) {
        cudaFree(d_pixels);
    }
    return true; // Success

error_exit:
    if (d_pixels) {
        cudaFree(d_pixels);
    }
    return false; // Failure
}