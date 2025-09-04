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
            static void ApplyBinarization(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // EdgeDetect
            static void ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyDifferentialSse(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplySobel(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplySobelSse(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyLaplacian(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyLaplacianSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // Blurring
            static void ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyAverageBlurSse(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyGaussianBlur(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyGaussianBlurSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // Morphorogy
            static void ApplyDilation(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyDilationSse(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyErosion(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyErosionSse(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // Image Matching
            static void ApplyNCC(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplySAD(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplySSD(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // TODO: Add other function declarations here later
        };
    }
}
