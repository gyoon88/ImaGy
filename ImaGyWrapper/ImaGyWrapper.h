#pragma once

// Include the native header
#include "..\ImaGyNative\NativeCore.h"
#include "..\ImaGyNative\NativeCoreSse.h"

// Reference .NET assemblies
#using <System.dll>
////#using <PresentationCore.dll>
//#using <WindowsBase.dll> // For BitmapSource

using namespace System;
using namespace System::Windows;
using namespace System::Windows::Media;
using namespace System::Windows::Media::Imaging;

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
            static void ApplyNCC(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);
            static void ApplySAD(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);
            static void ApplySSD(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);
        };
    }
}
