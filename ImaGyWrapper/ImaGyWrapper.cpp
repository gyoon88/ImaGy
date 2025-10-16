#include "pch.h"
#include "ImaGyWrapper.h"

// Allows managed code to get a native pointer to the underlying buffer of a managed array.
#include <vcclr.h>

namespace ImaGy
{
    namespace Wrapper
    {
        // Color Contrast
        void NativeProcessor::ApplyAdjBrightness(IntPtr pixels, int width, int height, int stride, int value)
        {
            ImaGyNative::NativeCore::ApplyAdjBrightness(pixels.ToPointer(), width, height, stride, value);
        }

        void NativeProcessor::ApplyBinarization(IntPtr pixels, int width, int height, int stride, int threshold)
        {
            ImaGyNative::NativeCore::ApplyBinarization(pixels.ToPointer(), width, height, stride, threshold);
        }

        void NativeProcessor::ApplyKMeansClustering(IntPtr pixels, int width, int height, int stride, int k, int iteration, bool location)
        {
            ImaGyNative::NativeCore::ApplyKMeansClustering(pixels.ToPointer(), width, height, stride, k, iteration, location);

        }
        void NativeProcessor::ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyEqualization(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyEqualizationColor(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyEqualizationColor(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplyHistogram(IntPtr pixels, int width, int height, int stride, int* hist)
        {
            ImaGyNative::NativeCore::ApplyHistogram(pixels.ToPointer(), width, height, stride, hist);
        }

        // Edge Detect
        void NativeProcessor::ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            ImaGyNative::NativeCore::ApplyDifferential(pixels.ToPointer(), width, height, stride, threshold);
        }
        void NativeProcessor::ApplySobel(IntPtr pixels, int width, int height, int stride, int kernelSize)
        {
            ImaGyNative::NativeCore::ApplySobel(pixels.ToPointer(), width, height, stride, kernelSize);
        }
        void NativeProcessor::ApplyLaplacian(IntPtr pixels, int width, int height, int stride, int kernelSize)
        {
            ImaGyNative::NativeCore::ApplyLaplacian(pixels.ToPointer(), width, height, stride, kernelSize);
        }


        void NativeProcessor::ApplyFFT(IntPtr pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase)
        {
            ImaGyNative::NativeCore::ApplyFFT(pixels.ToPointer(), width, height, stride, kernelSize, isInverse, isCPU, isPhase);
        }
        void NativeProcessor::ApplyFrequencyFilter(IntPtr pixels, int width, int height, int stride, int filterType, double radius)
        {
            ImaGyNative::NativeCore::ApplyFrequencyFilter(pixels.ToPointer(), width, height, stride, filterType, radius);

        }


        void NativeProcessor::ApplyAxialBandStopFilter(IntPtr pixels, int width, int height, int stride, double lowFreqRadius, double bandThickness)
        {
            ImaGyNative::NativeCore::ApplyAxialBandStopFilter(pixels.ToPointer(), width, height, stride, lowFreqRadius, bandThickness);
        }

        void NativeProcessor::ApplyFFTColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase)
        {
            ImaGyNative::NativeCore::ApplyFFTColor(pixels.ToPointer(), width, height, stride, kernelSize, isInverse, isCPU, isPhase);
        }

        // Blurring
        void NativeProcessor::ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyAverageBlur(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }
        void NativeProcessor::ApplyAverageBlurColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyAverageBlurColor(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }

        void NativeProcessor::ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyGaussianBlur(pixels.ToPointer(), width, height, stride, sigma, kernelSize, useCircularKernel);
        }
        void NativeProcessor::ApplyGaussianBlurColor(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyGaussianBlurColor(pixels.ToPointer(), width, height, stride, sigma, kernelSize, useCircularKernel);
        }


        // Morphorogy
        void NativeProcessor::ApplyDilation(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyDilation(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }
        void NativeProcessor::ApplyDilationColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyDilationColor(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }

        void NativeProcessor::ApplyErosion(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyErosion(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }
        void NativeProcessor::ApplyErosionColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
        {
            ImaGyNative::NativeCore::ApplyErosionColor(pixels.ToPointer(), width, height, stride, kernelSize, useCircularKernel);
        }


        // Image Matching
        void NativeProcessor::ApplyNCC(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords)
        {
            ImaGyNative::NativeCore::ApplyNCC(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, (int*)outCoords.ToPointer());
        }
        void NativeProcessor::ApplySAD(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords)
        {
            ImaGyNative::NativeCore::ApplySAD(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, (int*)outCoords.ToPointer());
        }
        void NativeProcessor::ApplySSD(IntPtr pixels, int width, int height, int stride, IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords)
        {
            ImaGyNative::NativeCore::ApplySSD(pixels.ToPointer(), width, height, stride, templatePixels.ToPointer(), templateWidth, templateHeight, templateStride, (int*)outCoords.ToPointer());
        }

    }
}