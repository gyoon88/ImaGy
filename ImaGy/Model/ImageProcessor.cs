using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Model
{
    public class ImageProcessor
    {
        public BitmapSource ApplyBinarization(BitmapSource source, byte threshold)
        {
            if (source == null) return null;

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
    }
}
