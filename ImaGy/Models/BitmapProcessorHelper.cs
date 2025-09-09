using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Models
{
    public static class BitmapProcessorHelper
    {
        /// <summary>
        /// Processes a BitmapSource by applying a native action to its pixel data.
        /// Automatically handles both grayscale and color images.
        /// </summary>
        public static BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            // If the image is already grayscale, process it directly.
            if (source.Format == PixelFormats.Gray8)
            {
                return ProcessGrayscaleImage(source, nativeAction);
            }
            // For all other formats (including color), process each color channel individually.
            else
            {
                return ProcessColorImage(source, nativeAction);
            }
        }

        /// <summary>
        /// Processes two images for matching. This method continues to operate on grayscale
        /// versions of the images, which is a common and effective approach for template matching.
        /// </summary>
        public static BitmapSource ProcessTwoBitmapSourcePixels(BitmapSource source, BitmapSource template, Action<IntPtr, int, int, int, IntPtr, int, int, int> nativeAction)
        {
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

            GCHandle pinnedSourcePixels = GCHandle.Alloc(sourcePixels, GCHandleType.Pinned);
            GCHandle pinnedTemplatePixels = GCHandle.Alloc(templatePixels, GCHandleType.Pinned);

            try
            {
                IntPtr sourcePixelPtr = pinnedSourcePixels.AddrOfPinnedObject();
                IntPtr templatePixelPtr = pinnedTemplatePixels.AddrOfPinnedObject();

                nativeAction(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride,
                             templatePixelPtr, templateWidth, templateHeight, templateStride);

                BitmapSource resultBitmap = BitmapSource.Create(
                    sourceWidth,
                    sourceHeight,
                    source.DpiX,
                    source.DpiY,
                    PixelFormats.Gray8,
                    null,
                    sourcePixels,
                    sourceStride);

                resultBitmap.Freeze();
                return resultBitmap;
            }
            finally
            {
                if (pinnedSourcePixels.IsAllocated) pinnedSourcePixels.Free();
                if (pinnedTemplatePixels.IsAllocated) pinnedTemplatePixels.Free();
            }
        }

        /// <summary>
        /// New logic for processing color images: splits into R, G, B channels, processes each, and merges back.
        /// </summary>
        private static BitmapSource ProcessColorImage(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            // 1. Standardize to Bgra32 format for consistent channel handling.
            var colorBitmap = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            int width = colorBitmap.PixelWidth;
            int height = colorBitmap.PixelHeight;
            int colorStride = width * 4;
            byte[] allPixels = new byte[height * colorStride];
            colorBitmap.CopyPixels(allPixels, colorStride, 0);

            // 2. Split pixel data into separate channels. Alpha is preserved but not processed.
            int channelSize = width * height;
            byte[] blueChannel = new byte[channelSize];
            byte[] greenChannel = new byte[channelSize];
            byte[] redChannel = new byte[channelSize];
            byte[] alphaChannel = new byte[channelSize];

            for (int i = 0; i < allPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                blueChannel[pixelIndex] = allPixels[i];
                greenChannel[pixelIndex] = allPixels[i + 1];
                redChannel[pixelIndex] = allPixels[i + 2];
                alphaChannel[pixelIndex] = allPixels[i + 3];
            }

            // 3. Process each color channel individually using the same native C++ function.
            ProcessSingleChannel(blueChannel, width, height, nativeAction);
            ProcessSingleChannel(greenChannel, width, height, nativeAction);
            ProcessSingleChannel(redChannel, width, height, nativeAction);

            // 4. Merge the processed channels back into a single pixel array.
            byte[] finalPixels = new byte[allPixels.Length];
            for (int i = 0; i < finalPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                finalPixels[i] = blueChannel[pixelIndex];
                finalPixels[i + 1] = greenChannel[pixelIndex];
                finalPixels[i + 2] = redChannel[pixelIndex];
                finalPixels[i + 3] = alphaChannel[pixelIndex]; // Restore original alpha
            }

            // 5. Create a new BitmapSource from the merged, processed pixel data.
            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, PixelFormats.Bgra32, null, finalPixels, colorStride);
            result.Freeze();
            return result;
        }

        /// <summary>
        /// Processes a single-channel (grayscale) image.
        /// </summary>
        private static BitmapSource ProcessGrayscaleImage(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int stride = (width * source.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[height * stride];
            source.CopyPixels(pixels, stride, 0);

            ProcessSingleChannel(pixels, width, height, nativeAction);

            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, source.Format, null, pixels, stride);
            result.Freeze();
            return result;
        }

        /// <summary>
        /// Helper that pins the memory of a byte array (a single channel) and executes the native action.
        /// </summary>
        private static void ProcessSingleChannel(byte[] channelPixels, int width, int height, Action<IntPtr, int, int, int> nativeAction)
        {
            int stride = width; // For a single channel, stride is always equal to its width.
            GCHandle pinnedPixels = GCHandle.Alloc(channelPixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject();
                nativeAction(pixelPtr, width, height, stride);
            }
            finally
            {
                pinnedPixels.Free();
            }
        }
    }
}
