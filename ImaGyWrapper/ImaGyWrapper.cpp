#include "pch.h"
#include "ImaGyWrapper.h"

// Allows managed code to get a native pointer to the underlying buffer of a managed array.
#include <vcclr.h>

namespace ImaGy
{
    namespace Wrapper
    {
        // Color Contrast
        void NativeProcessor::ApplyBinarization(IntPtr pixels, int width, int height, int stride, Byte threshold)
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
        void NativeProcessor::ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyAverageBlur(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyGaussianBlur(pixels.ToPointer(), width, height, stride, threshold);
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
        void NativeProcessor::ApplyNCC(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyNCC(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplySAD(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplySAD(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplySSD(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplySSD(pixels.ToPointer(), width, height, stride, threshold);
        }
    }
}