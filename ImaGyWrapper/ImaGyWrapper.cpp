#include "pch.h"
#include "ImaGyWrapper.h"

// Allows managed code to get a native pointer to the underlying buffer of a managed array.
#include <vcclr.h>

namespace ImaGy
{
    namespace Wrapper
    {
        void NativeProcessor::ApplyBinarization(IntPtr pixels, int width, int height, int stride, Byte threshold)
        {
            // Cast the IntPtr to void* and pass it directly to the native C++ function.
            ImaGyNative::NativeCore::ApplyBinarization(pixels.ToPointer(), width, height, stride, threshold);
            ImaGyNative::NativeCore::ApplyEqualization(pixels.ToPointer(), width, height, stride, threshold);
            
            ImaGyNative::NativeCore::ApplyDifferential(pixels.ToPointer(), width, height, stride, threshold);
            ImaGyNative::NativeCore::ApplyAverageBlur(pixels.ToPointer(), width, height, stride, threshold);
        }
    }
}
