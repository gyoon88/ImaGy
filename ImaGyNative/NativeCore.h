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
    };
}
