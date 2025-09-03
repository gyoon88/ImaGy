#pragma once

namespace ImaGyNative
{
    namespace SSE
    {
        void ApplyAverageBlurSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplyDilationSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplyErosionSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplyGaussianBlurSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplyLaplacianSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplyDifferentialSse(void* pixels, int width, int height, int stride, unsigned char threshold);
        void ApplySobelSse(void* pixels, int width, int height, int stride, unsigned char threshold);
    }
}
