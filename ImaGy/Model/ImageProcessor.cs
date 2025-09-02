using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Model
{
    public class ImageProcessor
    {
        // Colour | contrast
        public BitmapSource ApplyBinarization(BitmapSource source, byte threshold)
        {
            // Convert to grayscale
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
            grayBitmap.BeginInit();
            grayBitmap.Source = source;
            grayBitmap.DestinationFormat = PixelFormats.Gray8;
            grayBitmap.EndInit();

            // Get pixel data
            int stride = (grayBitmap.PixelWidth * grayBitmap.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[grayBitmap.PixelHeight * stride];
            grayBitmap.CopyPixels(pixels, stride, 0);

            // Apply threshold
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = pixels[i] < threshold ? (byte)0 : (byte)255;
            }

            // Create new bitmap
            BitmapSource result = BitmapSource.Create(
                grayBitmap.PixelWidth,
                grayBitmap.PixelHeight,
                grayBitmap.DpiX,
                grayBitmap.DpiY,
                grayBitmap.Format,
                null,
                pixels,
                stride);

            result.Freeze(); // Improve performance
            return result;
        }

        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        // BitmapSource To Memory for C++ Engine
        private BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int, byte> nativeAction, byte threshold)
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
                null, // palette
                pixels,  // Use the modified managed array
                stride);

            result.Freeze(); // Recommended for performance and thread safety
            return result;
        }


        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source)
        {
            return source;
        }

        public BitmapSource ApplySobel(BitmapSource source)
        {
            return source;
        }

        public BitmapSource ApplyLaplacian(BitmapSource source)
        {
            return source;
        }


        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source)
        {
            return source;
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source)
        {
            return source;
        }


        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source)
        {
            return source;

        }

        public BitmapSource ApplyErosion(BitmapSource source)
        {
            return source;
        }


        // Image Matching
        public BitmapSource ApplyNCC(BitmapSource source)
        {
            return source;
        }

        public BitmapSource ApplySAD(BitmapSource source)
        {
            return source;
        }

        public BitmapSource ApplySSD(BitmapSource source)
        {
            return source;
        }
    }

}
