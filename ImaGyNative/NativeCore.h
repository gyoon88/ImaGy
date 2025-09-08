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
        static void ApplyBinarization(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplyEqualization(void* pixels, int width, int height, int stride, unsigned char threshold);
        
        // Edge Detection
        static void ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplySobel(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplyLaplacian (void* pixels, int width, int height, int stride, unsigned char threshold);
        
        // Blurring
        static void ApplyAverageBlur(void* pixels, int width, int height, int stride, int kernelSize);
        static void ApplyGaussianBlur(void* pixels, int width, int height, int stride, double sigma, int kernelSize);

        // Morphorogy
        static void ApplyDilation(void* pixels, int width, int height, int stride, unsigned char threshold);
        static void ApplyErosion(void* pixels, int width, int height, int stride, unsigned char threshold);

        // Image Matching
        static void ApplyNCC(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold);
        static void ApplySAD(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold);
        static void ApplySSD(void* pixels, int width, int height, int stride, 
            void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold);


    };
}
