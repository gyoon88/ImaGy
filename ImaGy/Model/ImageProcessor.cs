using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;

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
            return source;
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
