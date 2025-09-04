using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Model
{
    public static class BitmapProcessorHelper
    {
        public static BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int, byte> nativeAction, byte threshold)
        {
            // Convert to grayscale if not already, as native function expects Gray8
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
            grayBitmap.BeginInit();
            grayBitmap.Source = source;
            grayBitmap.DestinationFormat = PixelFormats.Gray8;
            grayBitmap.EndInit();

            int width = grayBitmap.PixelWidth;
            int height = grayBitmap.PixelHeight;
            int stride = (width * grayBitmap.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[height * stride];
            grayBitmap.CopyPixels(pixels, stride, 0);

            GCHandle pinnedPixels = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject();
                nativeAction(pixelPtr, width, height, stride, threshold);
            }
            finally
            {
                pinnedPixels.Free();
            }

            // Create a new BitmapSource from the modified pixel data
            BitmapSource result = BitmapSource.Create(
                width,
                height,
                grayBitmap.DpiX,
                grayBitmap.DpiY,
                grayBitmap.Format,
                null,
                pixels,
                stride);

            result.Freeze();
            return result;
        }
        public static BitmapSource ProcessTwoBitmapSourcePixels(BitmapSource source, BitmapSource template, Action<IntPtr, int,
     int, int, IntPtr, int, int, int, byte> nativeAction, byte threshold)
        {
            // 1. Process Source Image
            // Convert source image to a usable format (Gray8)
            FormatConvertedBitmap graySourceBitmap = new FormatConvertedBitmap();
            graySourceBitmap.BeginInit();
            graySourceBitmap.Source = source;
            graySourceBitmap.DestinationFormat = PixelFormats.Gray8;
            graySourceBitmap.EndInit();

            int sourceWidth = graySourceBitmap.PixelWidth;
            int sourceHeight = graySourceBitmap.PixelHeight;
            int sourceStride = (sourceWidth * graySourceBitmap.Format.BitsPerPixel + 7) / 8;
            byte[] sourcePixels = new byte[sourceHeight * sourceStride];
            graySourceBitmap.CopyPixels(sourcePixels, sourceStride, 0);

            // 2. Process Template Image
            // Convert template image to a usable format (Gray8)
            FormatConvertedBitmap grayTemplateBitmap = new FormatConvertedBitmap();
            grayTemplateBitmap.BeginInit();
            grayTemplateBitmap.Source = template;
            grayTemplateBitmap.DestinationFormat = PixelFormats.Gray8;
            grayTemplateBitmap.EndInit();

            int templateWidth = grayTemplateBitmap.PixelWidth;
            int templateHeight = grayTemplateBitmap.PixelHeight;
            int templateStride = (templateWidth * grayTemplateBitmap.Format.BitsPerPixel + 7) / 8;
            byte[] templatePixels = new byte[templateHeight * templateStride];
            grayTemplateBitmap.CopyPixels(templatePixels, templateStride, 0);

            // Pin memory to get stable pointers for unmanaged code
            GCHandle pinnedSourcePixels = GCHandle.Alloc(sourcePixels, GCHandleType.Pinned);
            GCHandle pinnedTemplatePixels = GCHandle.Alloc(templatePixels, GCHandleType.Pinned);

            try
            {
                IntPtr sourcePixelPtr = pinnedSourcePixels.AddrOfPinnedObject();
                IntPtr templatePixelPtr = pinnedTemplatePixels.AddrOfPinnedObject();

                // The nativeAction modifies the sourcePixels array in-place.
                // This design assumes that the native C++ code will directly alter the memory
                // pointed to by sourcePixelPtr. No BitmapSource is returned from native code;
                // instead, we create a new one here from the modified byte array.
                nativeAction(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride,
                             templatePixelPtr, templateWidth, templateHeight, templateStride, threshold);

                // Recreate BitmapSource from the modified sourcePixels array.
                // This is the correct way to get the data back from the in-place modification.
                BitmapSource resultBitmap = BitmapSource.Create(
                    sourceWidth,
                    sourceHeight,
                    source.DpiX,
                    source.DpiY,
                    PixelFormats.Gray8, // Assuming the native code outputs Gray8 format
                    null,
                    sourcePixels,
                    sourceStride);

                // Freeze the bitmap for performance benefits and to make it thread-safe.
                resultBitmap.Freeze();
                return resultBitmap;
            }
            finally
            {
                // Always free the pinned handles to prevent memory leaks.
                if (pinnedSourcePixels.IsAllocated)
                {
                    pinnedSourcePixels.Free();
                }
                if (pinnedTemplatePixels.IsAllocated)
                {
                    pinnedTemplatePixels.Free();
                }
            }
        }
    }
}
