#pragma once
#include <vector> // std::vector�� ����ϱ� ���� �߰�

// C++ �ڵ忡�� ȣ���� CUDA ��ó �Լ����� �����մϴ�.
// ��� �Լ��� ���� �� true, ���� �� false�� ��ȯ�մϴ�.

// Pixel-wise
bool LaunchBinarizationKernel(unsigned char* pixels, int width, int height, int stride, int threshold);
bool LaunchEqualizationKernel(unsigned char* pixels, int width, int height, int stride);

// Filters (Shared Memory ����ȭ ����)
bool LaunchGaussianBlurKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
bool LaunchAverageBlurKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
bool LaunchSobelKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);
bool LaunchLaplacianKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);

// Morphology (Shared Memory ����ȭ ����)
bool LaunchDilationKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
bool LaunchErosionKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

// Template Matching
bool LaunchNccKernel(const unsigned char* image, int width, int height, int stride,
    const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
    int* out_x, int* out_y);

// FFT Filter
bool LaunchFftFilterKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);

