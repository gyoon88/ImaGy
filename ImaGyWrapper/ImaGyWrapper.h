#pragma once

// Include the native header
#include "..\ImaGyNative\NativeCore.h"

// Reference .NET assemblies
#using <System.dll>
//#using <PresentationCore.dll>
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
            // Takes a .NET BitmapSource, processes it using native C++, and returns a new BitmapSource
            static void ApplyBinarization(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyEqualization(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyDifferential(IntPtr pixels, int width, int height, int stride, Byte threshold);
            static void ApplyAverageBlur(IntPtr pixels, int width, int height, int stride, Byte threshold);

            // TODO: Add other function declarations here later
        };
    }
}
