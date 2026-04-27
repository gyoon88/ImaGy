#pragma once

// Include the native header
#include "..\ImaGyNative\NativeCore.h"
#include "..\ImaGyNative\NativeCoreSse.h"

// Reference .NET assemblies
#using <System.dll>
// WPF types: add when needed, e.g. #using <PresentationCore.dll> and the relevant namespaces.

using namespace System;

namespace ImaGy
{
    namespace Wrapper
    {
        public ref class NativeProcessor
        {
        public:
            // // Color Contrast
            static void ApplyAdjBrightness(IntPtr pixels, int width, int height, int stride, int value);

            static void ApplyBinarization(IntPtr pixels, int width, int height, int stride, int threshold);
            static void ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyEqualizationColor(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyKMeansClustering(IntPtr pixels, int width, int height, int stride, int k, int iteration, bool location);

            static void ApplyHistogram(IntPtr pixels, int width, int height, int stride, int* hist);

            // EdgeDetect
            static void ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplySobel(IntPtr pixels, int width, int height, int stride, int kernelSize);
            static void ApplyLaplacian(IntPtr pixels, int width, int height, int stride, int kernelSize);
            static void ApplyFFT(IntPtr pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase);
            static void ApplyFrequencyFilter(IntPtr pixels, int width, int height, int stride, int filterType, double radius);
            static void ApplyAxialBandStopFilter(IntPtr pixels, int width, int height, int stride, double lowFreqRadius, double bandThickness);

            static void ApplyFFTColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool isInverse, bool isCPU, bool isPhase);

            // Blurring
            static void ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyAverageBlurColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            
            static void ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
            static void ApplyGaussianBlurColor(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

            // Morphorogy
            static void ApplyDilation(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyDilationColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

            static void ApplyErosion(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyErosionColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);


            // Image Matching
            static void ApplyNCC(IntPtr pixels, int width, int height, int stride, 
                IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords);
            static void ApplySAD(IntPtr pixels, int width, int height, int stride, 
                IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords);
            static void ApplySSD(System::IntPtr pixels, int width, int height, int stride, 
                IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, IntPtr outCoords);
        };
    }
}
