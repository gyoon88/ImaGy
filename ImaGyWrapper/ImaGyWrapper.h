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
            static void ApplyBinarization(IntPtr pixels, int width, int height, int stride, int threshold);
            static void ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold);

            static void ApplyHistogram(IntPtr pixels, int width, int height, int stride, int* hist);

            // EdgeDetect
            static void ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyDifferentialSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            static void ApplySobel(IntPtr pixels, int width, int height, int stride, int kernelSize);
            static void ApplySobelSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            static void ApplyLaplacian(IntPtr pixels, int width, int height, int stride, int kernelSize);
            static void ApplyLaplacianSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // Blurring
            static void ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyAverageBlurColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            
            static void ApplyAverageBlurSse(IntPtr pixels, int width, int height, int stride, int kernelSize);

            static void ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
            static void ApplyGaussianBlurColor(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

            static void ApplyGaussianBlurSse(IntPtr pixels, int width, int height, int stride, double sigma, int kernelSize);

            // Morphorogy
            static void ApplyDilation(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyDilationColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

            static void ApplyDilationSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            static void ApplyErosion(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
            static void ApplyErosionColor(IntPtr pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

            static void ApplyErosionSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // Image Matching
            static void ApplyNCC(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);
            static void ApplySAD(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);
            static void ApplySSD(System::IntPtr pixels, int width, int height, int stride, 
                System::IntPtr templatePixels, int templateWidth, int templateHeight, int templateStride, System::IntPtr outCoords);




            // TODO: Add other function declarations here later
        };
    }
}
