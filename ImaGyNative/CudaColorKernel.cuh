
#pragma once

namespace ImaGyNative
{
    // ==========================================
    // --- CUDA 컬러 이미지 처리 Launcher 함수 (선언) ---
    // ==========================================

    // 가우시안 블러 (컬러)
    bool LaunchGaussianBlurColorKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

    // 평균 필터 (컬러)
    bool LaunchAverageBlurColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

    // 팽창 (컬러)
    bool LaunchDilationColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

    // 침식 (컬러)
    bool LaunchErosionColorKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

    // 히스토그램 평활화 (컬러)
    bool LaunchEqualizationColorKernel(unsigned char* pixels, int width, int height, int stride);

    // FFT 스펙트럼 (컬러)
    bool LaunchFftSpectrumColorKernel(unsigned char* pixels, int width, int height, int stride);
}