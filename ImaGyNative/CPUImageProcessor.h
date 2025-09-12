#pragma once

#include <vector>
namespace ImaGyNative
{
    static void ApplyConvolution(const unsigned char* sourcePixels, unsigned char* destPixels,
        int width, int height, int stride, const std::vector<double>& kernel, int kernelSize);


    static void ApplyConvolutionColor(const unsigned char* sourcePixels, unsigned char* destPixels,
        int width, int height, int stride, const std::vector<double>& kernel, int kernelSize);

    static void ApplyBinarization_CPU(void* pixels, int width, int height, int stride, int threshold);
    static void ApplyEqualization_CPU(void* pixels, int width, int height, int stride, unsigned char threshold);


    static void ApplyDifferential_CPU(void* pixels, int width, int height, int stride, unsigned char threshold);
    static void ApplySobel_CPU(void* pixels, int width, int height, int stride, int kernelSize);
    static void ApplyLaplacian_CPU(void* pixels, int width, int height, int stride, int kernelSize);

    static void ApplyGaussianBlur_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
    static void ApplyAverageBlur_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

    static void ApplyDilation_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
    static void ApplyErosion_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

    static void ApplyNCC_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight,
        int templateStride, int* outCoords);
    static void ApplySAD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);
    static void ApplySSD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);

    /////
    // Color
    // CPU Processing
    // 
    // Blur
    static void ApplyAverageBlurColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
    static void ApplyGaussianBlurColor_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

    //Morphology
    static void ApplyDilationColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
    static void ApplyErosionColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

}