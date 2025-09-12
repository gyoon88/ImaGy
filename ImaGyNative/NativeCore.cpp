// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include "ImageProcessingUtils.h"
#include "CPUImageProcessor.h"
#include "CudaKernel.cuh" // Include our new CUDA header
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 
#include <numeric>
#include <algorithm>
#include <stdexcept> // 예외 처리를 위해 추가
#include <cuda_runtime.h>



namespace ImaGyNative
{

    bool IsCudaAvailable() {
        static bool initialized = false;
        static bool is_available = false;

        if (!initialized) {
            int deviceCount = 0;
            cudaError_t err = cudaGetDeviceCount(&deviceCount);
            is_available = (err == cudaSuccess && deviceCount > 0);
            initialized = true;
        }
        return is_available;
    }

    // // Color Contrast

    // Histogram - Complete
    void NativeCore::ApplyHistogram(void* pixels, int width, int height, int stride, int* hist) {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        std::fill(hist, hist + 256, 0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = pixelData[y * stride + x];
                hist[idx]++;
            }
        }
    }

    void NativeCore::ApplyBinarization(void* pixels, int width, int height, int stride, int threshold)
    {
        if (IsCudaAvailable()) {
            if (LaunchBinarizationKernel(static_cast<unsigned char*>(pixels), width, height, stride, threshold)) {
                return; // CUDA 성공 시 종료
            }
        }
        // CUDA 실패 또는 사용 불가 시 CPU 코드 실행
        ApplyBinarization_CPU(pixels, width, height, stride, threshold);
    }

    // Equalization
    void NativeCore::ApplyEqualization(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        if (IsCudaAvailable()) {
            if (LaunchEqualizationKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
                return;
            }
        }
        ApplyEqualization_CPU(pixels, width, height, stride, threshold);
    }

    // Gaussian Blur
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchGaussianBlurKernel(static_cast<unsigned char*>(pixels), width, height, stride, sigma, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyGaussianBlur_CPU(pixels, width, height, stride, sigma, kernelSize, useCircularKernel);
    }

    // Average Blur
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchAverageBlurKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyAverageBlur_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }


    // // EdgeDetect
// Differnetial - Complete
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        ApplyDifferential_CPU( pixels,  width,  height,  stride,   threshold);
    }

    // Sobel
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (IsCudaAvailable()) {
            if (LaunchSobelKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize)) {
                return;
            }
        }
        ApplySobel_CPU(pixels, width, height, stride, kernelSize);
    }

    // Laplacian
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (IsCudaAvailable()) {
            if (LaunchLaplacianKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize)) {
                return;
            }
        }
        ApplyLaplacian_CPU(pixels, width, height, stride, kernelSize);
    }

    // Dilation
    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchDilationKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyDilation_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    // Erosion
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchErosionKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyErosion_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    // NCC
    void NativeCore::ApplyNCC(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
    {
        if (IsCudaAvailable()) {
            if (LaunchNccKernel(static_cast<const unsigned char*>(pixels), width, height, stride, static_cast<const unsigned char*>(templatePixels), templateWidth, templateHeight, templateStride, &outCoords[0], &outCoords[1])) {
                return;
            }
        }
        ApplyNCC_CPU(pixels, width, height, stride, templatePixels, templateWidth, templateHeight, templateStride, outCoords);
    }
    void NativeCore::ApplySAD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
    {
        ApplySAD_CPU(pixels, width, height, stride, templatePixels, templateWidth, templateHeight, templateStride, outCoords);
    }

    void NativeCore::ApplySSD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
    {
        ApplySSD_CPU(pixels, width, height, stride, templatePixels, templateWidth, templateHeight, templateStride, outCoords);
    }


    void NativeCore::ApplyFFT(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (IsCudaAvailable()) {
            // 새로 만든 스펙트럼 생성 함수를 호출
            if (LaunchFftSpectrumKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
                return;
            }
        }
    }

    void NativeCore::ApplyGaussianBlurColor(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
    {
        ApplyGaussianBlurColor_CPU(pixels, width, height, stride, sigma, kernelSize, useCircularKernel);
    }

    void NativeCore::ApplyAverageBlurColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        ApplyAverageBlurColor_CPU( pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    void NativeCore::ApplyDilationColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        ApplyDilationColor_CPU( pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    void NativeCore::ApplyErosionColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        ApplyErosionColor_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }

}
