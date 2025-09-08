#pragma once

#ifdef IMAGYNATIVE_EXPORTS
#define IMAGYNATIVE_API __declspec(dllexport)
#else
#define IMAGYNATIVE_API __declspec(dllimport)
#endif

namespace ImaGyNative
{
    namespace SSE
    {
        extern "C" IMAGYNATIVE_API void ApplyLaplacianSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        extern "C" IMAGYNATIVE_API void ApplyDifferentialSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        extern "C" IMAGYNATIVE_API void ApplySobelSse(void* pixels, int width, int height, int stride, unsigned char threshold);

        extern "C" IMAGYNATIVE_API void ApplyAverageBlurSse(void* pixels, int width, int height, int stride, int kernelSize);
        extern "C" IMAGYNATIVE_API void ApplyGaussianBlurSse(void* pixels, int width, int height, int stride, double sigma, int kernelSize);

        extern "C" IMAGYNATIVE_API void ApplyDilationSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        extern "C" IMAGYNATIVE_API void ApplyErosionSse(void* pixels, int width, int height, int stride, unsigned char threshold);
    }
}
