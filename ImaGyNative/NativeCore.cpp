// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include "ImageProcessingUtils.h"
#include "CPUImageProcessor.h"
#include "CudaKernel.cuh" 
#include "CudaColorKernel.cuh"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 
#include <numeric>
#include <algorithm>
#include <stdexcept> // 예외 처리
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
    if (threshold == -1)
    {
        threshold = OtsuThreshold(static_cast<unsigned char*>(pixels), width, height, stride);
    }
    if (IsCudaAvailable()) {
        if (LaunchBinarizationKernel(static_cast<unsigned char*>(pixels), width, height, stride, threshold)) {
            return; // CUDA 성공 시 종료
        }
    }
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
void NativeCore::ApplyEqualizationColor(void* pixels, int width, int height, int stride, unsigned char threshold)
{
    if (IsCudaAvailable()) {
        if (LaunchEqualizationColorKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
            return;
        }
    }
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
    ApplyDifferential_CPU(pixels, width, height, stride, threshold);
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

void NativeCore::ApplyFFT(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase)
{
    if (isCPU) {
        // FFT 연산을 위한 임시 복소수 배열 할당 
        Complex* tempSpectrum = new Complex[width * height];
        if (isPhase) {
            ApplyFFT2DPhase_CPU(pixels, tempSpectrum, width, height, stride, isInverse);
        }
        else {
            ApplyFFT2DSpectrum_CPU(pixels, tempSpectrum, width, height, stride, isInverse);
        }
        delete[] tempSpectrum;
    }
    else {
        if (IsCudaAvailable()) {
            if (LaunchFftSpectrumKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
                return;
            }
            Complex* tempSpectrum = new Complex[width * height];
            ApplyFFT2DSpectrum_CPU(pixels, tempSpectrum, width, height, stride, isInverse);
        }
    }
}


    void NativeCore::ApplyFFTColor(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase )
    {
        if (IsCudaAvailable()) {
            if (LaunchFftSpectrumColorKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
                return;
            }
        }
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
        if (IsCudaAvailable()) {
            if (LaunchSadKernel(static_cast<const unsigned char*>(pixels), width, height, stride, static_cast<const unsigned char*>(templatePixels), templateWidth, templateHeight, templateStride, &outCoords[0], &outCoords[1])) {
                return;
            }
        }
        ApplySAD_CPU(pixels, width, height, stride, templatePixels, templateWidth, templateHeight, templateStride, outCoords);
    }

    void NativeCore::ApplySSD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
    {
        if (IsCudaAvailable()) {
            if (LaunchSsdKernel(static_cast<const unsigned char*>(pixels), width, height, stride, static_cast<const unsigned char*>(templatePixels), templateWidth, templateHeight, templateStride, &outCoords[0], &outCoords[1])) {
                return;
            }
        }
        ApplySSD_CPU(pixels, width, height, stride, templatePixels, templateWidth, templateHeight, templateStride, outCoords);
    }



    /// <summary>
    /// Color 이미지의 가우스 블러를 처리하는 함수 
    /// </summary>
    /// <param name="pixels"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="stride"></param>
    /// <param name="sigma"></param>
    /// <param name="kernelSize"></param>
    /// <param name="useCircularKernel"></param>
    void NativeCore::ApplyGaussianBlurColor(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchGaussianBlurColorKernel(static_cast<unsigned char*>(pixels), width, height, stride, sigma, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyGaussianBlurColor_CPU(pixels, width, height, stride, sigma, kernelSize, useCircularKernel);
    }

    /// <summary>
    /// 컬러이미지의 평균 블러를 처리하는 함수
    /// GPU 호출 실패시 CPU 코드로 FallBack
    /// </summary>
    /// <param name="pixels">이미지가 있는 메모리 주소 </param>
    /// <param name="width">이미지 </param>
    /// <param name="height"></param>
    /// <param name="stride"></param>
    /// <param name="kernelSize"></param>
    /// <param name="useCircularKernel"></param>
    void NativeCore::ApplyAverageBlurColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            // 새로 만든 컬러 CUDA 함수를 호출
            if (LaunchAverageBlurColorKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel) ){
                return;
            }
        }
        ApplyAverageBlurColor_CPU( pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    /// <summary>
    /// 컬러 이미지 팽창을 처리하는 함수 
    /// GPU 호출 실패시 CPU 코드로 FallBack
    /// </summary>
    /// <param name="pixels"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="stride"></param>
    /// <param name="kernelSize"></param>
    /// <param name="useCircularKernel"></param>
    void NativeCore::ApplyDilationColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            // 새로 만든 컬러 CUDA 함수를 호출
            if (LaunchDilationColorKernel(static_cast<unsigned char*>(pixels), width, height, stride,  kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyDilationColor_CPU( pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    /// <summary>
    /// 컬러 이미지 팽창을 처리하는 함수 
    /// GPU 호출 실패시 CPU 코드로 FallBack
    /// </summary>
    /// <param name="pixels"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="stride"></param>
    /// <param name="kernelSize"></param>
    /// <param name="useCircularKernel"></param>
    void NativeCore::ApplyErosionColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchErosionColorKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyErosionColor_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    void NativeCore::ApplyKMeansClustering(void* pixels, int width, int height, int stride, int k, int iteration)
    {

        ApplyKMeansClusteringXY_Normalized_CPU(pixels,  width, height, stride, k, iteration);
    }

}
