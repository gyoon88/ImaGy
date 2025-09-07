using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public static class ImageChannelUtils
    {
        public static (byte[] r, byte[] g, byte[] b) SplitChannels(BitmapSource source)
        {
            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int stride = width * 4;
            byte[] pixels = new byte[height * stride];
            source.CopyPixels(pixels, stride, 0);

            byte[] r = new byte[width * height];
            byte[] g = new byte[width * height];
            byte[] b = new byte[width * height];

            for (int i = 0; i < width * height; i++)
            {
                int index = i * 4;
                b[i] = pixels[index + 0];
                g[i] = pixels[index + 1];
                r[i] = pixels[index + 2];
            }

            return (r, g, b);
        }

        public static BitmapSource CombineChannels(byte[] r, byte[] g, byte[] b, int width, int height)
        {
            int stride = width * 4;
            byte[] pixels = new byte[height * stride];

            for (int i = 0; i < width * height; i++)
            {
                int index = i * 4;
                pixels[index + 0] = b[i];
                pixels[index + 1] = g[i];
                pixels[index + 2] = r[i];
                pixels[index + 3] = 255; // Alpha
            }

            return BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, pixels, stride);
        }

        public static BitmapSource ToGrayscale(BitmapSource source)
        {
            return new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
        }
    }
}
