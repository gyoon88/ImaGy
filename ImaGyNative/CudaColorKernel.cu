// 파일 이름: CudaColorKernel.cu

#include "CudaColorKernel.cuh"
#include "CudaKernel.cuh" // 흑백용 FFT 커널 등을 재사용하기 위해 포함
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cufft.h>

namespace ImaGyNative
{
    // --- Helper Macro & Constant Memory ---
#define CUDA_CHECK(err_code) do { cudaError_t _err = (err_code); if (_err != cudaSuccess) { return false; } } while (0)
#define CUFFT_CHECK(err_code) do { cufftResult _err = (err_code); if (_err != CUFFT_SUCCESS) { return false; } } while (0)

// 가우시안, 평균 필터 커널을 저장하기 위한 상수 메모리
    __constant__ float c_colorFilterKernel[625];

    // CPU에서 float 타입의 커널을 생성하는 헬퍼 함수 (흑백용 코드에서 가져옴)
    std::vector<float> createGaussianKernelFloat(int kernelSize, double sigma, bool isCircular);
    std::vector<float> createAverageKernelFloat(int kernelSize, bool isCircular);

    // ==========================================
    // --- CUDA 컬러 커널 정의 ---
    // ==========================================

    // 컬러 컨볼루션 커널 (가우시안, 평균 필터)
    __global__ void ColorConvolutionKernel(const uchar4* input, uchar4* output, int width, int height, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int center = kernelSize / 2;
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;

        for (int ky = -center; ky <= center; ++ky) {
            for (int kx = -center; kx <= center; ++kx) {
                int nX = x + kx;
                int nY = y + ky;

                if (nX >= 0 && nX < width && nY >= 0 && nY < height) {
                    float weight = c_colorFilterKernel[(ky + center) * kernelSize + (kx + center)];
                    uchar4 pixel = input[nY * width + nX];
                    sumB += pixel.x * weight; // .x는 B
                    sumG += pixel.y * weight; // .y는 G
                    sumR += pixel.z * weight; // .z는 R
                }
            }
        }

        uchar4 outPixel;
        outPixel.x = (unsigned char)fmaxf(0.f, fminf(255.f, sumB));
        outPixel.y = (unsigned char)fmaxf(0.f, fminf(255.f, sumG));
        outPixel.z = (unsigned char)fmaxf(0.f, fminf(255.f, sumR));
        outPixel.w = input[y * width + x].w; // Alpha 채널은 보존

        output[y * width + x] = outPixel;
    }

    // 컬러 팽창 커널
    __global__ void ColorDilationKernel(const uchar4* input, uchar4* output, int width, int height, int kernelSize, bool useCircularKernel) // 파라미터 추가
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int center = kernelSize / 2;
        unsigned char maxB = 0, maxG = 0, maxR = 0;

        for (int ky = -center; ky <= center; ++ky) {
            for (int kx = -center; kx <= center; ++kx) {
                // ✨ 추가된 부분: 원형 커널일 경우, 원 밖의 픽셀은 건너뜁니다.
                if (useCircularKernel && (kx * kx + ky * ky) > (center * center)) {
                    continue;
                }

                int nX = x + kx;
                int nY = y + ky;
                if (nX >= 0 && nX < width && nY >= 0 && nY < height) {
                    uchar4 pixel = input[nY * width + nX];
                    maxB = max(maxB, pixel.x);
                    maxG = max(maxG, pixel.y);
                    maxR = max(maxR, pixel.z);
                }
            }
        }
        output[y * width + x] = make_uchar4(maxB, maxG, maxR, input[y * width + x].w);
    }

    // 컬러 침식 커널
    __global__ void ColorErosionKernel(const uchar4* input, uchar4* output, int width, int height, int kernelSize, bool useCircularKernel) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int center = kernelSize / 2;
        unsigned char minB = 255, minG = 255, minR = 255;

        for (int ky = -center; ky <= center; ++ky) {
            for (int kx = -center; kx <= center; ++kx) {
                // 원형 커널일 경우, 원 밖의 픽셀은 건너뜀
                if (useCircularKernel && (kx * kx + ky * ky) > (center * center)) {
                    continue;
                }

                int nX = x + kx;
                int nY = y + ky;
                if (nX >= 0 && nX < width && nY >= 0 && nY < height) {
                    uchar4 pixel = input[nY * width + nX];
                    minB = min(minB, pixel.x);
                    minG = min(minG, pixel.y);
                    minR = min(minR, pixel.z);
                }
            }
        }
        output[y * width + x] = make_uchar4(minB, minG, minR, input[y * width + x].w);
    }

    // BGRA -> 3채널(B,G,R)로 분리하는 커널
    __global__ void SplitBGRAKernel(const uchar4* input, unsigned char* b_out, unsigned char* g_out, unsigned char* r_out, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            uchar4 pixel = input[i];
            b_out[i] = pixel.x;
            g_out[i] = pixel.y;
            r_out[i] = pixel.z;
        }
    }

    // 3채널(B,G,R) -> BGRA로 병합하는 커널
    __global__ void MergeToBGRAKernel(unsigned char* bgra_out, const unsigned char* b_in, const unsigned char* g_in, const unsigned char* r_in, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            // uchar4*로 캐스팅하여 4바이트 단위로 쓰기
            ((uchar4*)bgra_out)[i] = make_uchar4(b_in[i], g_in[i], r_in[i], 255);
        }
    }


    // ==========================================
    // --- CUDA 컬러 Launcher 함수 구현 ---
    // ==========================================

    bool LaunchGaussianBlurColorKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel) {
        if (stride != width * 4) return false;

        size_t imageSize = (size_t)height * stride;
        uchar4* d_input = nullptr, * d_output = nullptr;
        std::vector<float> h_kernel = createGaussianKernelFloat(kernelSize, sigma, useCircularKernel);

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_colorFilterKernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        ColorConvolutionKernel << <grid, block >> > ((uchar4*)d_input, (uchar4*)d_output, width, height, kernelSize);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        return true;
    }

    bool LaunchAverageBlurColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
        // LaunchGaussianBlurColorKernel과 거의 동일하고 h_kernel 생성 부분만 다름
        if (stride != width * 4) return false;

        size_t imageSize = (size_t)height * stride;
        uchar4* d_input = nullptr, * d_output = nullptr;
        std::vector<float> h_kernel = createAverageKernelFloat(kernelSize, useCircularKernel);

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_colorFilterKernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        ColorConvolutionKernel << <grid, block >> > ((uchar4*)d_input, (uchar4*)d_output, width, height, kernelSize);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        return true;
    }

    bool LaunchDilationColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
        if (stride != width * 4) return false;
        size_t imageSize = (size_t)height * stride;
        uchar4* d_input = nullptr, * d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        ColorDilationKernel << <grid, block >> > ((uchar4*)d_input, (uchar4*)d_output, width, height, kernelSize, useCircularKernel);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        return true;
    }

    bool LaunchErosionColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
        if (stride != width * 4) return false;
        size_t imageSize = (size_t)height * stride;
        uchar4* d_input = nullptr, * d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        ColorErosionKernel << <grid, block >> > ((uchar4*)d_input, (uchar4*)d_output, width, height, kernelSize, useCircularKernel);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        return true;
    }

    bool LaunchEqualizationColorKernel(unsigned char* pixels, int width, int height, int stride) {
        if (stride != width * 4) return false;
        const int N = width * height;
        size_t imageSize = (size_t)height * stride;
        size_t channelSize = N * sizeof(unsigned char);

        uchar4* d_input = nullptr;
        unsigned char* d_b = nullptr, * d_g = nullptr, * d_r = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_b, channelSize));
        CUDA_CHECK(cudaMalloc(&d_g, channelSize));
        CUDA_CHECK(cudaMalloc(&d_r, channelSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

        // BGRA -> B, G, R 세 채널로 분리
        dim3 grid1D((N + 255) / 256);
        dim3 block1D(256);
        SplitBGRAKernel << <grid1D, block1D >> > ((uchar4*)d_input, d_b, d_g, d_r, N);

        // 각 채널에 흑백용 평활화 함수 재사용
        LaunchEqualizationKernel(d_b, width, height, width);
        LaunchEqualizationKernel(d_g, width, height, width);
        LaunchEqualizationKernel(d_r, width, height, width);

        // 처리된 B, G, R 채널을 다시 BGRA로 병합
        MergeToBGRAKernel << <grid1D, block1D >> > ((unsigned char*)d_input, d_b, d_g, d_r, N);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_input, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_b);
        cudaFree(d_g);
        cudaFree(d_r);
        return true;
    }

    bool LaunchFftSpectrumColorKernel(unsigned char* pixels, int width, int height, int stride) {
        if (stride != width * 4) return false;
        const int N = width * height;
        size_t imageSize = (size_t)height * stride;
        size_t channelSize = N * sizeof(unsigned char);

        uchar4* d_input = nullptr;
        unsigned char* d_b = nullptr, * d_g = nullptr, * d_r = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_b, channelSize));
        CUDA_CHECK(cudaMalloc(&d_g, channelSize));
        CUDA_CHECK(cudaMalloc(&d_r, channelSize));
        CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

        // 1. BGRA -> B, G, R 채널 분리
        dim3 grid1D((N + 255) / 256);
        dim3 block1D(256);
        SplitBGRAKernel << <grid1D, block1D >> > ((uchar4*)d_input, d_b, d_g, d_r, N);

        // 각 채널에 흑백용 FFT 스펙트럼 함수 재사용
        LaunchFftSpectrumKernel(d_b, width, height, width);
        LaunchFftSpectrumKernel(d_g, width, height, width);
        LaunchFftSpectrumKernel(d_r, width, height, width);

        // 처리된 B, G, R 채널을 다시 BGRA로 병합
        MergeToBGRAKernel << <grid1D, block1D >> > ((unsigned char*)d_input, d_b, d_g, d_r, N);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels, d_input, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_b);
        cudaFree(d_g);
        cudaFree(d_r);
        return true;
    }
}