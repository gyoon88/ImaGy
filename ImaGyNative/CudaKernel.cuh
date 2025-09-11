#pragma once
#include <vector> // std::vector를 사용하기 위해 추가

// C++ 코드에서 호출할 CUDA 런처 함수들을 선언합니다.
// 모든 함수는 성공 시 true, 실패 시 false를 반환합니다.

// Pixel-wise
bool LaunchBinarizationKernel(unsigned char* pixels, int width, int height, int stride, int threshold);
bool LaunchEqualizationKernel(unsigned char* pixels, int width, int height, int stride);

// Filters (Shared Memory 최적화 적용)
bool LaunchGaussianBlurKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
bool LaunchAverageBlurKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
bool LaunchSobelKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);
bool LaunchLaplacianKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);

// Morphology (Shared Memory 최적화 적용)
bool LaunchDilationKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
bool LaunchErosionKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

// Template Matching
bool LaunchNccKernel(const unsigned char* image, int width, int height, int stride,
    const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
    int* out_x, int* out_y);

// FFT Filter
bool LaunchFftFilterKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);

