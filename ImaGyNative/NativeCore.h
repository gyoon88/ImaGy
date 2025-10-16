// NativeCore.h
#pragma once

#ifdef IMAGYNATIVE_EXPORTS
#define IMAGYNATIVE_API __declspec(dllexport)
#else
#define IMAGYNATIVE_API __declspec(dllimport)
#endif

namespace ImaGyNative
{
    class IMAGYNATIVE_API NativeCore
    {
    public:
        // Applies binarization to grayscale pixel data.
        static void ApplyAdjBrightness(void* pixels, int width, int height, int stride, int value);
        static void ApplyBinarization(void* pixels, int width, int height, int stride, int threshold);

        static void ApplyEqualization(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplyEqualizationColor(void* pixels, int width, int height, int stride, unsigned char threshold);
        
        static void ApplyKMeansClustering(void* pixels, int width, int height, int stride, int k, int iteration, bool location);

        static void ApplyHistogram(void* pixels, int width, int height, int stride, int* hist);

        // Edge Detection
        static void ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplySobel(void* pixels, int width, int height, int stride, int kernelSize);
        static void ApplyLaplacian (void* pixels, int width, int height, int stride, int kernelSize);

        static void ApplyFFT(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase);
        static void ApplyFFTColor(void* pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase);

        static void ApplyFrequencyFilter(void* pixels, int width, int height, int stride, int filterType, double radius);
        static void ApplyAxialBandStopFilter(void* pixels, int width, int height, int stride, double lowFreqRadius, double bandThickness);

        // Blurring
        static void ApplyGaussianBlur(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
        static void ApplyGaussianBlurColor(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

        static void ApplyAverageBlur(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
        static void ApplyAverageBlurColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

        // Morphorogy
        static void ApplyDilation(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
        static void ApplyDilationColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

        static void ApplyErosion(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
        static void ApplyErosionColor(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);


        // Image Matching
                // Image Matching
        static void ApplyNCC(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);
        static void ApplySAD(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);
        static void ApplySSD(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);
    };
}
