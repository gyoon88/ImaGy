#include "pch.h"
#include "ImaGyWrapper.h"

// Allows managed code to get a native pointer to the underlying buffer of a managed array.
#include <vcclr.h>

namespace ImaGy
{
    namespace Wrapper
    {
        // Color Contrast
        void NativeProcessor::ApplyBinarization(IntPtr pixels, int width, int height, int stride, int threshold)
        {
            ImaGyNative::NativeCore::ApplyBinarization(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyEqualization(pixels.ToPointer(), width, height, stride, threshold);
        }


        // Edge Detect
        void NativeProcessor::ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyDifferential(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplySobel(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplySobel(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyLaplacian(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyLaplacian(pixels.ToPointer(), width, height, stride, threshold);
        }

        // Blurring
        void NativeProcessor::ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, int kernelSize)
        {
            ImaGyNative::NativeCore::ApplyAverageBlur(pixels.ToPointer(), width, height, stride, kernelSize);
        }

        void NativeProcessor::ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize)
        {
            ImaGyNative::NativeCore::ApplyGaussianBlur(pixels.ToPointer(), width, height, stride, sigma, kernelSize);
        }


        // Morphorogy
        void NativeProcessor::ApplyDilation(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyDilation(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyErosion(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyErosion(pixels.ToPointer(), width, height, stride, threshold);
        }
        
        // Image Matching
        void NativeProcessor::ApplyNCC(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
        {
            ImaGyNative::NativeCore::ApplyNCC(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, threshold);
        }

        void NativeProcessor::ApplySAD(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
        {
            ImaGyNative::NativeCore::ApplySAD(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, threshold);
        }

        void NativeProcessor::ApplySSD(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
        {
            ImaGyNative::NativeCore::ApplySSD(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, threshold);
        }



        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // SSE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Edge Detect
        void NativeProcessor::ApplyDifferentialSse(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::SSE::ApplyDifferentialSse(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplySobelSse(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::SSE::ApplySobelSse(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyLaplacianSse(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::SSE::ApplyLaplacianSse(pixels.ToPointer(), width, height, stride, threshold);
        }

        // Blurring
        void NativeProcessor::ApplyAverageBlurSse(IntPtr pixels, int width, int height, int stride, int kernelSize)
        {
            ImaGyNative::SSE::ApplyAverageBlurSse(pixels.ToPointer(), width, height, stride, kernelSize);
        }

        void NativeProcessor::ApplyGaussianBlurSse(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize)
        {
            ImaGyNative::SSE::ApplyGaussianBlurSse(pixels.ToPointer(), width, height, stride, sigma, kernelSize);
        }

        // Morphorogy
        void NativeProcessor::ApplyDilationSse(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::SSE::ApplyDilationSse(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyErosionSse(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::SSE::ApplyErosionSse(pixels.ToPointer(), width, height, stride, threshold);
        }
    }
}
