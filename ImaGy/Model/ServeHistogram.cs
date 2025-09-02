using System.Windows.Media.Imaging;
using System.Windows.Media;

namespace ImaGy.Model
{
    public class ServeHistogram
    {
        public static int[] CalculateGrayscaleHistogram(BitmapSource source)
        {
            if (source == null) return new int[256];

            // Convert to grayscale if not already
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
            grayBitmap.BeginInit();
            grayBitmap.Source = source;
            grayBitmap.DestinationFormat = PixelFormats.Gray8;
            grayBitmap.EndInit();

            int[] histogram = new int[256];
            int stride = grayBitmap.PixelWidth;
            byte[] pixels = new byte[grayBitmap.PixelHeight * stride];
            grayBitmap.CopyPixels(pixels, stride, 0);

            foreach (byte pixelValue in pixels)
            {
                histogram[pixelValue]++;
            }

            return histogram;
        }
    }
}
