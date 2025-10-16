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
    // Check the GPU
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

    void NativeCore::ApplyAdjBrightness(void* pixels, int width, int height, int stride, int value){
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

#pragma omp parallel for
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * stride + j;
                // 픽셀 값 + value가 0~255 범위를 벗어나지 않도록 클램핑(clamping)
                int newValue = static_cast<int>(pixelData[index]) + value;
                pixelData[index] = static_cast<unsigned char>(std::max(0, std::min(255, newValue)));
            }
        }
    }

    /// Color Contrast
    // Histogram - Complete
    void NativeCore::ApplyHistogram(void* pixels, int width, int height, int stride, int* hist) {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        std::fill(hist, hist + 256, 0); // 히스토그램 0으로 초기화

        // #pragma omp parallel 블록으로 병렬 영역을 생성
    #pragma omp parallel
        {
            // 각 스레드가 사용할 자신만의 로컬 히스토그램을 생성

            int local_hist[256] = { 0 };

            // #pragma omp for로 루프를 스레드에 분배
            //    각 스레드는 자신만의 local_hist에만 값을 더함.
    #pragma omp for nowait
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    local_hist[pixelData[y * stride + x]]++;
                }
            }

            // 모든 스레드의 계산이 끝나면, #pragma omp critical을 사용해
            //    한 번에 하나의 스레드만 접근하도록 하여 결과를 안전하게 합침.
    #pragma omp critical
            for (int i = 0; i < 256; ++i) {
                hist[i] += local_hist[i];
            }
        }
    }

    // Video Segmentation
    void NativeCore::ApplyBinarization(void* pixels, int width, int height, int stride, int threshold)
    {
        if (threshold == -1)
        {
            threshold = OtsuThreshold(static_cast<unsigned char*>(pixels), width, height, stride);
        }
        if (IsCudaAvailable()) {
            if (LaunchBinarizationKernel(static_cast<unsigned char*>(pixels), width, height, stride, threshold)) {
                return; 
            }
        }
        ApplyBinarization_CPU(pixels, width, height, stride, threshold);
    }
    void NativeCore::ApplyKMeansClustering(void* pixels, int width, int height, int stride, int k, int iteration, bool location)
    {
        if (location) {
            ApplyKMeansClusteringXY_Normalized_CPU(pixels, width, height, stride, k, iteration);

        }
        else {
            ApplyKMeansClustering_CPU(pixels, width, height, stride, k, iteration);

        }
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

    /// Filtering
    // EdgeDetect    
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        ApplyDifferential_CPU(pixels, width, height, stride, threshold);
    }
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (IsCudaAvailable()) {
            if (LaunchSobelKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize)) {
                return;
            }
        }
        ApplySobel_CPU(pixels, width, height, stride, kernelSize);
    }
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (IsCudaAvailable()) {
            if (LaunchLaplacianKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize)) {
                return;
            }
        }
        ApplyLaplacian_CPU(pixels, width, height, stride, kernelSize);
    }

    // Blur
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchGaussianBlurKernel(static_cast<unsigned char*>(pixels), width, height, stride, sigma, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyGaussianBlur_CPU(pixels, width, height, stride, sigma, kernelSize, useCircularKernel);
    }
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchAverageBlurKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyAverageBlur_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
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
    /// <param name="width">이미지 넓이</param>
    /// <param name="height">이미지 높이</param>
    /// <param name="stride">픽셀당 바이트</param>
    /// <param name="kernelSize">커널 생성 지름 또는 한변의 길이</param>
    /// <param name="useCircularKernel">원형 커널 생성 여부</param>
    void NativeCore::ApplyAverageBlurColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchAverageBlurColorKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyAverageBlurColor_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }

    // FFT 
    void NativeCore::ApplyFFT(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase)
    {
        if (isCPU) {
            // FFT 연산을 위한 임시 복소수 배열
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
                if (isPhase) {
                    ApplyFFT2DPhase_CPU(pixels, tempSpectrum, width, height, stride, isInverse);
                }
                else {
                    ApplyFFT2DSpectrum_CPU(pixels, tempSpectrum, width, height, stride, isInverse);
                }
            }
        }
    }
    // frequency blocking
    void NativeCore::ApplyFrequencyFilter(void* pixels, int width, int height, int stride, int filterType,  double radius) {
        
        FilterType ft = static_cast<FilterType>(filterType);
        ApplyFrequencyFilter_CPU(pixels, width, height, stride, ft, radius);
    }

    void NativeCore::ApplyAxialBandStopFilter(void* pixels, int width, int height, int stride, double lowFreqRadius, double bandThickness) {
        ApplyAxialBandStopFilter_CPU(pixels, width, height, stride, lowFreqRadius, bandThickness);
    }

    void NativeCore::ApplyFFTColor(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase)
    {
        if (IsCudaAvailable()) {
            if (LaunchFftSpectrumColorKernel(static_cast<unsigned char*>(pixels), width, height, stride)) {
                return;
            }
        }
    }
    // Morphology
    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchDilationKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyDilation_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
    }
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (IsCudaAvailable()) {
            if (LaunchErosionKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyErosion_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
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
            if (LaunchDilationColorKernel(static_cast<unsigned char*>(pixels), width, height, stride, kernelSize, useCircularKernel)) {
                return;
            }
        }
        ApplyDilationColor_CPU(pixels, width, height, stride, kernelSize, useCircularKernel);
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

    /// NCC
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



}
